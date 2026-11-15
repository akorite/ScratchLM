// Browser smoke check: load the text SB3 and loader.js unsandboxed in TurboWarp,
// select a real .bsp, then paste this script in DevTools. It sends Hi, observes
// actual layer progress, and stops inference. No model output is mocked.
(async () => {
  const root = document.querySelector('#bonsai-chat')?.shadowRoot;
  const assert = (condition, message) => { if (!condition) throw new Error(message); };
  assert(root, 'Load the text project and updated loader extension first');
  const send = root.querySelector('.send');
  assert(!send.disabled, 'Select a valid model package before running this check');
  const input = root.querySelector('textarea');
  input.value = 'Hi';
  root.querySelector('form').requestSubmit();
  assert(send.disabled, 'Send must be disabled during inference');
  assert(!root.querySelector('.stop').hidden, 'Stop must be available during inference');
  const deadline = performance.now() + 180000;
  try {
    while (!/layer \d+ \/ 24/.test(root.querySelector('.phase').textContent)) {
      assert(performance.now() < deadline, 'No live layer progress within three minutes');
      assert(!root.querySelector('.stop').hidden, 'Generation ended before reaching a layer');
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    const bar = root.querySelector('progress');
    assert(!bar.hidden && bar.max === 24 && bar.value >= 0 && bar.value < 24,
      'Layer progress must use the actual 24-layer bound');
    assert(/Prompt token \d+ of \d+/.test(root.querySelector('.detail').textContent),
      'Prompt progress must show actual token counts');
    assert(root.querySelector('.message.user').textContent.includes('Hi'),
      'The sent prompt must appear in the conversation');
    root.querySelector('.stop').click();
    assert(root.querySelector('.phase').textContent === 'Stopped', 'Stop must update status');
    assert(root.querySelector('.stop').hidden && bar.hidden, 'Stopped work must not look active');
    assert(!send.disabled && !input.disabled, 'Stopping must restore the composer');
    console.log('Chat UI smoke check passed: send, real layers, token counts, stop, composer reset.');
    return true;
  } finally {
    if (!root.querySelector('.stop').hidden) root.querySelector('.stop').click();
  }
})()
