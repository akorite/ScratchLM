#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Capture {
    std::string dir, filter;
    std::ofstream index, staging;
    size_t serial = 0, stage_serial = 0;
    int step = 0;
    bool failed = false;
};

static Capture * active_capture = nullptr;

extern "C" void bonsai_capture_q8k(const ggml_tensor * node, const void * data, int64_t elements) {
    if (!active_capture || active_capture->filter.empty() ||
        std::string(node->name).find(active_capture->filter) == std::string::npos) return;
    auto & c = *active_capture;
    if (elements <= 0 || elements % QK_K) { c.failed = true; return; }
    const auto * blocks = static_cast<const block_q8_K *>(data);
    const std::string file = "stage-" + std::to_string(c.stage_serial++) + ".bin";
    std::ofstream out(c.dir + "/" + file, std::ios::binary);
    for (int64_t i = 0; i < elements / QK_K; ++i) {
        out.write(reinterpret_cast<const char *>(&blocks[i].d), sizeof(blocks[i].d));
        out.write(reinterpret_cast<const char *>(blocks[i].qs), sizeof(blocks[i].qs));
    }
    out.close();
    c.staging << c.step << '\t' << node->name << '\t' << ggml_op_name(node->op) << '\t'
              << node->src[0]->name << '\t' << node->src[1]->name << '\t' << elements << '\t' << file << '\n';
    if (!out || !c.staging) c.failed = true;
}

static bool write_bytes(const std::string & path, const void * data, size_t size) {
    std::ofstream out(path, std::ios::binary);
    out.write(static_cast<const char *>(data), size);
    return bool(out);
}

static bool quantizable(const std::vector<float> & values) {
    for (size_t i = 0; i < values.size(); i += QK_K) {
        float max = 0;
        for (size_t j = i; j < i + QK_K; ++j) {
            if (!std::isfinite(values[j])) return false;
            max = std::max(max, std::fabs(values[j]));
        }
        if (max != 0 && !std::isfinite(127.0f / max)) return false;
    }
    return true;
}

static void capture_tensor(Capture & c, const ggml_tensor * t, const ggml_tensor * node, const std::string & role) {
    if (!t || !t->data || ggml_is_quantized(t->type)) return;
    const auto id = std::to_string(c.serial++);
    const std::string file = "tensor-" + id + ".bin";
    std::vector<uint8_t> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t, bytes.data(), 0, bytes.size());
    if (!write_bytes(c.dir + "/" + file, bytes.data(), bytes.size())) { c.failed = true; return; }
    std::string qfile = "-";
    if (t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] % QK_K == 0) {
        std::vector<float> values(ggml_nelements(t));
        std::memcpy(values.data(), bytes.data(), bytes.size());
        if (quantizable(values)) {
            std::vector<block_q8_K> q(values.size() / QK_K);
            quantize_row_q8_K_ref(values.data(), q.data(), values.size());
            qfile = "tensor-" + id + ".q8_k";
            if (!write_bytes(c.dir + "/" + qfile, q.data(), q.size() * sizeof(block_q8_K))) { c.failed = true; return; }
        }
    }
    c.index << c.step << '\t' << node->name << '\t' << ggml_op_name(node->op) << '\t' << role << '\t' << t->name << '\t' << ggml_type_name(t->type) << '\t';
    for (int i = 0; i < 4; ++i) c.index << (i ? "," : "") << t->ne[i];
    c.index << '\t';
    for (int i = 0; i < 4; ++i) c.index << (i ? "," : "") << t->nb[i];
    c.index << '\t' << file << '\t' << qfile << '\n';
    if (!c.index) c.failed = true;
}

static bool callback(ggml_tensor * t, bool ask, void * data) {
    auto & c = *static_cast<Capture *>(data);
    const bool selected = std::string(t->name).find(c.filter) != std::string::npos;
    if (ask) return selected;
    if (selected) {
        capture_tensor(c, t, t, "out");
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            if (t->src[i]) capture_tensor(c, t->src[i], t, "src" + std::to_string(i));
        }
    }
    return !c.failed;
}

static int quantize_file(const char * input, const char * output) {
    std::ifstream in(input, std::ios::binary | std::ios::ate);
    const std::streamoff size = in.tellg();
    if (!in || size <= 0 || size % (QK_K * sizeof(float)) != 0) {
        std::cerr << "input must contain complete 256-element F32 groups\n"; return 1;
    }
    std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
    in.seekg(0);
    in.read(reinterpret_cast<char *>(values.data()), size);
    if (!in || !quantizable(values)) {
        std::cerr << "input has nonfinite values or a group whose Q8_K scale overflows F32\n"; return 1;
    }
    std::vector<block_q8_K> q(values.size() / QK_K);
    quantize_row_q8_K_ref(values.data(), q.data(), values.size());
    if (!write_bytes(output, q.data(), q.size() * sizeof(block_q8_K))) return 1;
    return 0;
}

static int project_pq2(const char * model, const char * offset_arg, const char * rows_arg,
                       const char * input, const char * output) {
    try {
        const uint64_t offset = std::stoull(offset_arg), rows = std::stoull(rows_arg);
        std::ifstream activations(input, std::ios::binary | std::ios::ate);
        if (!activations) return 1;
        const auto bytes = activations.tellg();
        if (bytes <= 0 || bytes % (256 * sizeof(float)) || bytes > 6144 * sizeof(float) ||
            rows == 0 || rows > 248320) return 1;
        const int width = int(bytes / sizeof(float));
        std::vector<float> x(width);
        activations.seekg(0);
        activations.read(reinterpret_cast<char *>(x.data()), bytes);
        if (!activations || !quantizable(x)) return 1;
        std::vector<block_q8_K> q(width / QK_K);
        quantize_row_q8_K_ref(x.data(), q.data(), width);
        const uint64_t row_bytes = uint64_t(width / QK_PQ2_0) * sizeof(block_pq2_0);
        std::ifstream weights(model, std::ios::binary | std::ios::ate);
        if (!weights) return 1;
        const uint64_t size = uint64_t(weights.tellg());
        if (offset > size || rows > (size - offset) / row_bytes) return 1;
        weights.seekg(offset);
        std::vector<uint8_t> packed(rows * row_bytes);
        weights.read(reinterpret_cast<char *>(packed.data()), packed.size());
        if (!weights) return 1;
        ggml_cpu_init();
        const auto * trait = ggml_get_type_traits_cpu(GGML_TYPE_PQ2_0);
        if (!trait || !trait->vec_dot || trait->vec_dot_type != GGML_TYPE_Q8_K) return 1;
        std::vector<float> result(rows);
        for (uint64_t row = 0; row < rows; ++row) {
            trait->vec_dot(width, &result[row], 0, packed.data() + row * row_bytes, 0, q.data(), 0, 1);
        }
        return write_bytes(output, result.data(), result.size() * sizeof(float)) ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "PQ2 projection: " << e.what() << '\n';
        return 1;
    }
}


int main(int argc, char ** argv) {
    if (argc == 7 && std::string(argv[1]) == "--project-pq2") return project_pq2(argv[2], argv[3], argv[4], argv[5], argv[6]);
    if (argc == 4 && std::string(argv[1]) == "--quantize") return quantize_file(argv[2], argv[3]);
    if (argc < 7) { std::cerr << "model output context generate filter token... required\n"; return 2; }
    const std::string model_path = argv[1], dir = argv[2], filter = argv[5];
    const int context = std::stoi(argv[3]), generate = std::stoi(argv[4]);
    std::vector<llama_token> tokens;
    for (int i = 6; i < argc; ++i) tokens.push_back(std::stoi(argv[i]));
    Capture capture{dir, filter};
    capture.index.open(dir + "/captures.tsv");
    capture.staging.open(dir + "/staging.tsv");
    if (!capture.staging) return 1;
    active_capture = &capture;
    if (!capture.index) return 1;
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { std::cerr << "model load failed\n"; return 1; }
    const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (std::any_of(tokens.begin(), tokens.end(), [vocab](int t) { return t < 0 || t >= vocab; })) {
        std::cerr << "token outside model vocabulary\n"; return 1;
    }
    auto cp = llama_context_default_params();
    cp.n_ctx = context; cp.n_batch = 1; cp.n_ubatch = 1;
    cp.n_threads = 1; cp.n_threads_batch = 1;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.offload_kqv = false; cp.op_offload = false;
    if (!filter.empty()) { cp.cb_eval = callback; cp.cb_eval_user_data = &capture; }
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::cerr << "context initialization failed\n"; return 1; }
    std::ofstream info(dir + "/native.tsv");
    info << "effective_context\t" << llama_n_ctx(ctx) << "\nvocab_size\t" << vocab << '\n';
    const size_t input_count = tokens.size();
    for (size_t step = 0; step < input_count + static_cast<size_t>(generate); ++step) {
        capture.step = step;
        auto batch = llama_batch_get_one(&tokens[step], 1);
        if (llama_decode(ctx, batch) != 0 || capture.failed) { std::cerr << "decode or capture failed\n"; return 1; }
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (!logits) { std::cerr << "logits unavailable\n"; return 1; }
        if (!write_bytes(dir + "/logits-" + std::to_string(step) + ".f32", logits, vocab * sizeof(float))) return 1;
        if (!std::all_of(logits, logits + vocab, [](float f) { return std::isfinite(f); })) { std::cerr << "nonfinite logits\n"; return 1; }
        int best = int(std::max_element(logits, logits + vocab) - logits);
        info << "step\t" << step << '\t' << tokens[step] << '\t' << best << '\n';
        const size_t size = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> state(size);
        size_t written = llama_state_seq_get_data(ctx, state.data(), state.size(), 0);
        if (!written || !write_bytes(dir + "/state-" + std::to_string(step) + ".bin", state.data(), written)) { std::cerr << "state capture unavailable\n"; return 1; }
        if (step + 1 >= input_count && step + 1 < input_count + static_cast<size_t>(generate)) tokens.push_back(best);
    }
    if (!filter.empty() && !capture.serial) { std::cerr << "capture filter matched no available tensors\n"; return 1; }
    if (!info || !capture.index || !capture.staging) return 1;
    llama_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
