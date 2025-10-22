const tabs = document.querySelectorAll('.anag-tabs button');
try { console.info('[anag-ui] app.js v35 loaded'); } catch {}
const panels = document.querySelectorAll('.tab-panel');
const uploadForm = document.getElementById('upload-form');
const ingestLog = document.getElementById('ingest-log');
const askForm = document.getElementById('ask-form');
const askLog = document.getElementById('ask-log');
const askProgress = document.getElementById('ask-progress');
const askStatus = document.getElementById('ask-status');
const imageInput = document.getElementById('image-input');
const attachBtn = document.getElementById('attach-image');
const attachList = document.getElementById('chat-attachments');
const pageViewer = document.getElementById('page-viewer');
const pageViewerImg = document.getElementById('page-viewer-img');
const summaryContent = document.getElementById('summary-content');
const snapshotContent = document.getElementById('snapshot-content');
const chatWindow = document.getElementById('chat-window');
const chatBody = document.getElementById('chat-body');
const scrollDownBtn = document.getElementById('scroll-down');
let bottomSentinel = null;
let sentinelObserver = null;
let isAtBottom = true; // updated by IntersectionObserver or math fallback
let hasSentinelSignal = false;
// Temporary spacer to pin last user message near the top without permanent space
let tempBottomSpacer = null;
let pinnedTopMessage = null;
// Holds HTML preview of the user's current message (e.g., image thumbs) until saved
let pendingUserHtml = null;
let chatAttachments = [];

function ensureBottomSentinel() {
  if (!chatWindow) return null;
  if (!bottomSentinel || !bottomSentinel.isConnected) {
    bottomSentinel = document.createElement('div');
    bottomSentinel.id = 'scroll-sentinel';
    bottomSentinel.style.cssText = 'height:1px;width:100%;';
    chatWindow.appendChild(bottomSentinel);
  } else if (chatWindow.lastElementChild !== bottomSentinel) {
    chatWindow.appendChild(bottomSentinel);
  }
  return bottomSentinel;
}
const clearChatBtn = document.getElementById('clear-chat');
const convListEl = document.getElementById('conversation-list');
const newChatBtn = document.getElementById('new-chat');
const sidebarToggle = document.getElementById('toggle-sidebar');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const controlsDetails = document.querySelector('.chat-controls');
let controlsIsCompact = false;
const libraryGrid = document.getElementById('library-grid');
const libraryEmpty = document.getElementById('library-empty');
const librarySearchInput = document.getElementById('library-search');
const librarySortSelect = document.getElementById('library-sort');
const refreshLibraryBtn = document.getElementById('refresh-library'); // may not exist
const onlyDocSelect = document.getElementById('only-doc');
const webToggle = document.getElementById('web-enabled');
const strictToggle = document.getElementById('strict-docs');
const saveLibraryBtn = document.getElementById('save-library');
const clearLibraryBtn = document.getElementById('clear-library');
const cancelLibraryBtn = document.getElementById('cancel-library');
const settingsForm = document.getElementById('settings-form');
const settingsStatus = document.getElementById('settings-status');
let settingsSaveTimer = null;
let settingsSaveController = null;
const uploadProgressTrack = document.getElementById('upload-progress-track');
const uploadProgressBar = document.getElementById('upload-progress');
const uploadProgressWrap = uploadProgressTrack ? uploadProgressTrack.parentElement : null;
const bootstrapEl = document.getElementById('bootstrap-settings');
const uploadListEl = document.getElementById('upload-list');
let selectedFiles = [];
const dropZone = document.querySelector('.ingest-dropzone');
const docModal = document.getElementById('doc-details-modal');

function showDocModal(html) {
  try {
    if (!docModal) return;
    const inner = docModal.querySelector('.doc-inner');
    if (inner) inner.innerHTML = html;
    docModal.style.display = 'flex';
    docModal.setAttribute('aria-hidden', 'false');
  } catch {}
}

function hideDocModal() {
  try {
    if (!docModal) return;
    docModal.style.display = 'none';
    docModal.setAttribute('aria-hidden', 'true');
  } catch {}
}

docModal?.addEventListener('click', (e) => {
  const target = e.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.classList.contains('doc-backdrop') || target.classList.contains('doc-close')) {
    hideDocModal();
  }
});

async function openDocDetails(name) {
  try {
    const resp = await fetch(`${API_BASE}/library/doc/${encodeURIComponent(name)}`, { credentials: 'include' });
    if (!resp.ok) {
      const t = await resp.text(); showError(t || 'Failed to load document'); return;
    }
    const d = await resp.json();
    const esc = escapeHTML;
    const headings = (Array.isArray(d.top_headings) && d.top_headings.length)
      ? '<ul>' + d.top_headings.map(h => `<li>${esc(h.heading)} <small>(${h.count})</small></li>`).join('') + '</ul>'
      : '<em>No headings found.</em>';
    const samples = (Array.isArray(d.samples) && d.samples.length)
      ? '<ul>' + d.samples.map(s => `<li>[p.${esc(String(s.page || '?'))}] ${esc(String(s.snippet || '')).slice(0, 300)}</li>`).join('') + '</ul>'
      : '<em>No samples available.</em>';
    const pages = Array.isArray(d.pages) ? d.pages.join(', ') : '';
    const html = `
      <div style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
        <h3 style="margin:0;">${esc(d.display_name || d.name)}</h3>
        <button type="button" class="doc-close" aria-label="Close" style="border:1px solid var(--accent-border); background:var(--accent-bg-mid); color:var(--accent-text); border-radius:10px; padding:6px 10px; font-weight:600;">Close</button>
      </div>
      <div style="margin-top:8px; color:#334155;">
        <div style="display:flex; gap:14px; flex-wrap:wrap;">
          <span><strong>Pages:</strong> ${esc(String(d.page_count || 0))}</span>
          <span><strong>Chunks:</strong> ${esc(String(d.chunk_count || 0))}</span>
          <span><strong>Math chunks:</strong> ${esc(String(d.math_chunks || 0))}</span>
          <span><strong>Equations:</strong> ${esc(String(d.equations || 0))}</span>
          <span><strong>Formula chunks:</strong> ${esc(String(d.formula_chunks || 0))}</span>
        </div>
        ${pages ? `<div style="margin-top:6px"><strong>Pages list:</strong> ${esc(String(pages))}</div>` : ''}
        <div style="margin-top:10px"><strong>Top headings</strong>${headings}</div>
        <div style="margin-top:10px"><strong>Samples</strong>${samples}</div>
        <div style="margin-top:14px; display:flex; gap:8px;">
          <button type="button" class="primary" id="ask-only-doc">Ask only this doc</button>
          <button type="button" class="secondary doc-close">Close</button>
        </div>
      </div>`;
    showDocModal(html);
    const askBtn = docModal.querySelector('#ask-only-doc');
    askBtn?.addEventListener('click', () => {
      try {
        if (onlyDocSelect) {
          onlyDocSelect.value = d.name || name;
        }
        switchTab('ask');
        hideDocModal();
        const q = document.getElementById('question');
        if (q) q.focus();
      } catch {}
    });
  } catch (err) {
    showError(err);
  }
}

const API_HOST = (window.location && window.location.hostname) ? window.location.hostname : 'localhost';
// Use same-origin /api when not running local dev server on 7860
const API_BASE = (window.location && (window.location.port === '7860' || window.location.hostname === 'localhost'))
  ? `http://${API_HOST}:8000/api`
  : `${window.location.origin}/api`;
const AUTH_BASE = `${API_BASE}/auth`;

// Global error banner helpers
let __errorHideTimer = null;
let __errorLastMsg = null;
let __errorVisible = false;
function showError(message) {
  try {
    const banner = document.getElementById('error-banner');
    const textEl = document.getElementById('error-banner-text');
    if (!banner || !textEl) return;
    const msg = (typeof message === 'string') ? message : (message && message.message) ? message.message : String(message || 'Error');
    const clipped = msg.slice(0, 800);
    const same = __errorVisible && __errorLastMsg === clipped;
    textEl.textContent = clipped;
    banner.classList.add('show');
    banner.setAttribute('aria-hidden', 'false');
    __errorVisible = true;
    __errorLastMsg = clipped;
    if (!same) {
      if (__errorHideTimer) { clearTimeout(__errorHideTimer); __errorHideTimer = null; }
      __errorHideTimer = setTimeout(() => {
        try { hideError(); } catch {}
      }, 5000);
    }
  } catch {}
}
function hideError() {
  try {
    const banner = document.getElementById('error-banner');
    if (!banner) return;
    banner.classList.remove('show');
    banner.setAttribute('aria-hidden', 'true');
    if (__errorHideTimer) { clearTimeout(__errorHideTimer); __errorHideTimer = null; }
    __errorVisible = false;
  } catch {}
}
document.getElementById('error-banner-close')?.addEventListener('click', hideError);

// Wrap fetch to surface any network/HTTP errors in the banner
(function wrapFetch(){
  if (!window.fetch || window.fetch.__wrapped) return;
  const native = window.fetch.bind(window);
  const wrapped = async function(input, init) {
    try {
      const resp = await native(input, init);
      try {
        if (!resp.ok) {
          const url = (typeof input === 'string') ? input : (input && input.url) ? input.url : '';
          const clone = resp.clone();
          let body = '';
          try { body = await clone.text(); } catch {}
          const details = (body || '').slice(0, 300).replace(/\s+/g, ' ').trim();
          showError(`${resp.status} ${resp.statusText} — ${url}${details ? ' — ' + details : ''}`);
        }
      } catch {}
      return resp;
    } catch (err) {
      showError(err && err.message ? err.message : String(err));
      throw err;
    }
  };
  wrapped.__wrapped = true;
  window.fetch = wrapped;
})();

// Global listeners for unexpected errors
window.addEventListener('error', (e) => {
  if (!e) return; showError(e.message || (e.error && e.error.message) || 'Unexpected error');
});
window.addEventListener('unhandledrejection', (e) => {
  try { const r = e && e.reason; showError(r && (r.message || String(r)) || 'Unhandled rejection'); } catch {}
});
const chatHistory = [];
let currentUserId = null;
try {
  const authEl = document.getElementById('bootstrap-auth');
  if (authEl?.textContent) {
    const authData = JSON.parse(authEl.textContent);
    const u = authData && authData.user;
    const uid = u && (u.id || u.user_id || u.email);
    if (uid) currentUserId = String(uid);
  }
} catch {}
// Bump the storage prefix to reset all existing chats for all users
const STORAGE_PREFIX = 'anag_conversations_v2';
const STORAGE_KEY = currentUserId ? `${STORAGE_PREFIX}_${currentUserId}` : STORAGE_PREFIX;
let conversations = {};
let activeConvId = null;
const ingestPolls = new Map();
let defaultsData = {};
const pendingRemovals = new Set();
let librarySaveController = null;
let __libraryData = [];
let __librarySearch = '';
let __librarySort = 'date';
let __libraryDir = 'desc'; // 'asc' or 'desc'

// Friendly starter titles shown only when a chat is empty (not persisted)
// Keep these as greetings/intent prompts rather than concrete examples.
const EMPTY_CHAT_TITLES = [
  "Welcome — ready to begin?",
  "Hello! What can I help with?",
  "How can I assist today?",
  "Good to see you — what’s next?",
  "Tell me what you’re working on",
  "What would you like to explore?",
  "Ask me anything to get started",
  "I’m ready when you are",
  "What are we learning today?",
  "Where should we begin?",
  "Need a hand with your notes?",
  "What topic is on your mind?",
  "Describe your goal — I’ll help",
  "What would you like to accomplish?",
  "Got a question? I’m listening",
  "What problem can we tackle?",
  "What’s your research question?",
  "What insight are you chasing?",
  "What should we summarize?",
  "Who or what should we compare?",
  "How can I speed up your work?",
  "What would you like explained?",
  "What should we draft together?",
  "Point me at a challenge",
  "What would you like to review?",
  "Hello — ask away",
  "Tell me your task — I’ll help",
  "How can I make this easier?",
  "What needs clarification?",
  "What should we break down?",
  "Let’s get started",
  "Ready when you are — ask a question",
  "What’s your first step today?",
  "What’s blocking you right now?",
  "What should we prioritize?",
  "Share context and I’ll dive in",
  "What shall we analyze?",
  "What would you like to automate?",
  "How would you like to proceed?",
  "What would you like to extract?",
  "Tell me what you need",
  "Ready for your first question",
  "What should we look up?",
  "What would you like to verify?",
  "What do you want to understand?",
  "Let me know what to do",
  "What would you like summarized?",
  "What can I organize for you?",
  "What shall we plan?",
  "Where do you want to focus?",
];

let lastEmptyTitle = null;

function normalizeDashes(text) {
  try {
    let s = String(text || '');
    // Replace any spaced em/en dash with a comma + single space
    s = s.replace(/\s*[—–]\s*/g, ', ');
    // Remove space(s) directly before commas
    s = s.replace(/\s+,/g, ',');
    // Ensure single space after comma when there is text following
    s = s.replace(/,(?!\s|$)/g, ', ');
    // Collapse multiple spaces
    s = s.replace(/\s{2,}/g, ' ');
    return s.trim();
  } catch {
    return String(text || '');
  }
}

function showEmptyChatTitle() {
  if (!chatWindow) return;
  if (chatWindow.querySelector('.chat-empty')) return;
  let pick = EMPTY_CHAT_TITLES[Math.floor(Math.random() * EMPTY_CHAT_TITLES.length)] || 'Start your new chat';
  // Avoid immediate repeats when reopening/clearing
  if (EMPTY_CHAT_TITLES.length > 1 && pick === lastEmptyTitle) {
    pick = EMPTY_CHAT_TITLES[(Math.floor(Math.random() * (EMPTY_CHAT_TITLES.length - 1)) + 1) % EMPTY_CHAT_TITLES.length];
  }
  lastEmptyTitle = pick;
  const el = document.createElement('div');
  el.className = 'chat-empty';
  el.textContent = normalizeDashes(pick);
  chatWindow.appendChild(el);
}

function hideEmptyChatTitle() {
  if (!chatWindow) return;
  const el = chatWindow.querySelector('.chat-empty');
  if (el) el.remove();
}

// Example prompts for the textarea placeholder. Rotate on init and when clicked/focused if empty.
const PLACEHOLDER_EXAMPLES = [
  'Summarize the main points of the latest paper',
  'Create a bullet outline for chapter 3',
  'Compare transformer and RNN approaches in my notes',
  'Extract all formulas from the PDF named “Signals.pdf”',
  'Explain vector embeddings like I’m five',
  'Write a study guide for linear algebra',
  'Find contradictions across the climate reports',
  'List definitions of “entropy” across sources',
  'Draft a short abstract of the oncology paper',
  'What are the key takeaways from the EU AI Act?',
  'Generate flashcards for reinforcement learning',
  'Locate figures that mention “ROC curve”',
  'Cross-reference mentions of “No Free Lunch”',
  'Translate the conclusion of the French article',
  'Summarize chapter 2 in 5 bullets',
  'Outline a presentation on diffusion models',
  'Compare BGE-M3 vs MiniLM reranking results',
  'What evidence supports claim X in the report?',
  'Find caveats or limitations in this paper',
  'Extract citations related to transfer learning',
  'Explain t-SNE vs UMAP and when to use each',
  'Generate a reading plan for week 1',
  'Summarize the methodology section concisely',
  'List datasets mentioned in the paper',
  'Find all references to “randomized control trial”',
  'Draft an email summarizing the attached notes',
  'Turn these meeting notes into action items',
  'Identify open questions raised by the authors',
  'What prior work do they cite most often?',
  'Summarize the limitations in one paragraph',
  'Extract equations and define each variable',
  'Create a glossary for this document',
  'Compare results across 2022 and 2023 reports',
  'What does the discussion section conclude?',
  'Outline an experiment based on the paper',
  'Find steps for reproducing the method',
  'Summarize differences between A and B',
  'Where do they define the loss function?',
  'Build a study checklist for the exam',
  'List the assumptions made in the model',
  'Find mentions of “confidence interval”',
  'Write a plain-language summary for non-experts',
  'Compare two sections: Methods vs Results',
  'Find figures that contradict the hypothesis',
  'Create an outline for a literature review',
  'Extract data tables and summarize them',
  'What problems remain unsolved here?',
  'Summarize what changed since the last version',
  'List pros and cons of the proposed approach',
  'Create 10 flashcards from the introduction',
  'Explain this theorem step by step',
];

let lastPlaceholderText = null;
function setRandomPlaceholder(force = false) {
  const q = document.getElementById('question');
  if (!q) return;
  if (!force && q.value && q.value.trim().length > 0) return;
  if (!Array.isArray(PLACEHOLDER_EXAMPLES) || PLACEHOLDER_EXAMPLES.length === 0) return;
  let pick = PLACEHOLDER_EXAMPLES[Math.floor(Math.random() * PLACEHOLDER_EXAMPLES.length)];
  if (PLACEHOLDER_EXAMPLES.length > 1 && pick === lastPlaceholderText) {
    pick = PLACEHOLDER_EXAMPLES[(Math.floor(Math.random() * (PLACEHOLDER_EXAMPLES.length - 1)) + 1) % PLACEHOLDER_EXAMPLES.length];
  }
  lastPlaceholderText = pick;
  q.setAttribute('placeholder', normalizeDashes(pick));
}

function updateLibraryActionButtons() {
  if (saveLibraryBtn) {
    if (pendingRemovals.size > 0) {
      saveLibraryBtn.style.display = '';
      saveLibraryBtn.textContent = `Save (${pendingRemovals.size})`;
    } else {
      saveLibraryBtn.style.display = 'none';
    }
  }
}

function scrollIngestToBottom() {
  try { if (ingestLog) ingestLog.scrollTop = ingestLog.scrollHeight; } catch {}
}

if (bootstrapEl?.textContent) {
  try {
    defaultsData = JSON.parse(bootstrapEl.textContent);
  } catch (err) {
    console.warn('Failed to parse bootstrap defaults', err);
  }
}

function setSidebarOpen(open) {
  if (open) {
    document.body.classList.add('sidebar-open');
  } else {
    document.body.classList.remove('sidebar-open');
  }
  if (sidebarToggle) {
    sidebarToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    sidebarToggle.classList.toggle('is-open', open);
  }
}

function toggleSidebar() {
  const isOpen = document.body.classList.contains('sidebar-open');
  setSidebarOpen(!isOpen);
}

const sidebarMedia = window.matchMedia('(max-width: 1100px)');
function handleSidebarViewportChange() {
  if (sidebarToggle) {
    if (sidebarMedia.matches) {
      sidebarToggle.style.display = 'inline-flex';
    } else {
      sidebarToggle.style.display = 'none';
    }
  }
  if (!sidebarMedia.matches) {
    setSidebarOpen(false);
  }
}

function updateControlsDrawer(force = false) {
  if (!controlsDetails) return;
  const compact = window.innerWidth <= 1100;
  if (compact !== controlsIsCompact || force) {
    controlsIsCompact = compact;
    if (compact) {
      controlsDetails.open = false;
    } else {
      controlsDetails.open = true;
    }
  }
}

sidebarMedia.addEventListener?.('change', handleSidebarViewportChange);
sidebarMedia.addListener?.(handleSidebarViewportChange);
sidebarToggle?.addEventListener('click', toggleSidebar);
sidebarBackdrop?.addEventListener('click', () => setSidebarOpen(false));

// Measure and pin CSS variables used by layout to avoid visible snapping
function measureLayoutVars() {
  const root = document.documentElement;
  try {
    const hdr = document.querySelector('.anag-header');
    if (hdr) root.style.setProperty('--header-h', Math.ceil(hdr.getBoundingClientRect().height) + 'px');
  } catch {}
  try {
    const ftr = document.querySelector('.chat-footer');
    if (ftr) root.style.setProperty('--footer-h', Math.ceil(ftr.getBoundingClientRect().height) + 'px');
  } catch {}
  try {
    const sdb = document.querySelector('.chat-sidebar');
    if (sdb) root.style.setProperty('--sidebar-w', Math.ceil(sdb.getBoundingClientRect().width) + 'px');
  } catch {}
}

// Ensure the scroll-to-bottom button sits just above the footer regardless
// of CSS var timing or dynamic footer height changes.
function positionScrollButton() {
  if (!scrollDownBtn) return;
  let offset = 16;
  try {
    const ftr = document.querySelector('.chat-footer');
    if (ftr) {
      const h = Math.ceil(ftr.getBoundingClientRect().height);
      if (h > 0) offset = h + 16;
    }
  } catch {}
  scrollDownBtn.style.bottom = offset + 'px';
}

function applyDefaults(defaults) {
  if (!defaults) return;
  defaultsData = defaults;
  const setValue = (id, value) => {
    const el = document.getElementById(id);
    if (el && value !== undefined && value !== null) {
      el.value = value;
    }
  };
  const setChecked = (id, value) => {
    const el = document.getElementById(id);
    if (el) {
      el.checked = Boolean(value);
    }
  };

  // Conversation memory is always enabled in chat; the toggle is removed.
  setValue('max-batches', defaults.ASK_MAX_BATCHES);
  setValue('time-budget', defaults.ASK_TIME_BUDGET_SEC);
  const rerankerEl = document.getElementById('reranker');
  if (rerankerEl) rerankerEl.value = defaults.ASK_RERANKER || 'off';
  setChecked('exhaustive', defaults.ASK_EXHAUSTIVE);

  setValue('openai-model', defaults.OPENAI_CHAT_MODEL);
  setValue('hf-model', defaults.HF_LLM_NAME);
  setValue('embed-backend', defaults.EMBED_BACKEND);
  setValue('llm-backend', defaults.LLM_BACKEND);
  setChecked('settings-memory', defaults.MEMORY_ENABLED);
  setValue('memory-tokens', defaults.MEMORY_TOKEN_LIMIT);
  setValue('memory-file-mb', defaults.MEMORY_FILE_LIMIT_MB);
  setValue('openai-tpm', defaults.OPENAI_TPM);
  setValue('openai-rpm', defaults.OPENAI_RPM);
  setValue('ask-char-budget', defaults.ASK_BATCH_CHAR_BUDGET);
  setValue('ask-max-batches', defaults.ASK_MAX_BATCHES);
  setValue('ask-time-budget', defaults.ASK_TIME_BUDGET_SEC);
  setChecked('settings-exhaustive', defaults.ASK_EXHAUSTIVE);
  const rerankerSettings = document.getElementById('settings-reranker');
  if (rerankerSettings) rerankerSettings.value = (defaults.ASK_RERANKER && defaults.ASK_RERANKER !== 'off') ? defaults.ASK_RERANKER : 'minilm';
  setValue('ask-candidates', defaults.ASK_CANDIDATES);
  setValue('web-provider', defaults.WEB_SEARCH_PROVIDER || 'auto');
  const rerOn = document.getElementById('reranker-on');
  // Default OFF regardless of server defaults
  if (rerOn) rerOn.checked = false;
  // web scraping controls removed
}

applyDefaults(defaultsData);

// Ensure Web search and Strict docs are mutually exclusive in the UI
if (webToggle) {
  webToggle.addEventListener('change', () => {
    try {
      if (webToggle.checked && strictToggle) {
        strictToggle.checked = false;
      }
    } catch {}
  });
}
if (strictToggle) {
  strictToggle.addEventListener('change', () => {
    try {
      if (strictToggle.checked && webToggle) {
        webToggle.checked = false;
      }
    } catch {}
  });
}

// Exhaustive popover: show settings only on demand when toggled on
const exPopover = document.getElementById('ex-popover');
const exCheckbox = document.getElementById('exhaustive');
function positionExPopover() {
  try {
    if (!exPopover || !exCheckbox || !exCheckbox.checked) return;
    const r = exCheckbox.getBoundingClientRect();
    const panel = exPopover.querySelector('.ex-panel');
    const pad = 8;
    const x = Math.max(8, Math.min((r.left + (r.width/2) - 160), window.innerWidth - 340));
    const y = Math.max(8, r.top - (panel?.offsetHeight || 120) - 12);
    exPopover.style.left = x + 'px';
    exPopover.style.top = y + 'px';
  } catch {}
}
function showExPopover(show) {
  try {
    if (!exPopover) return;
    exPopover.style.display = show ? 'block' : 'none';
    if (show) {
      positionExPopover();
      // Focus first field
      const first = exPopover.querySelector('input');
      try { first?.focus(); } catch {}
      window.addEventListener('resize', positionExPopover);
      window.addEventListener('scroll', positionExPopover, true);
      document.addEventListener('click', handleExOutside, true);
    } else {
      window.removeEventListener('resize', positionExPopover);
      window.removeEventListener('scroll', positionExPopover, true);
      document.removeEventListener('click', handleExOutside, true);
    }
  } catch {}
}
function handleExOutside(e){
  try {
    if (!exPopover || exPopover.style.display === 'none') return;
    const t = e.target;
    if (!(t instanceof Node)) return;
    if (!exPopover.contains(t) && !exCheckbox.contains(t)) {
      showExPopover(false);
    }
  } catch {}
}
exCheckbox?.addEventListener('change', () => {
  try {
    if (exCheckbox.checked) {
      showExPopover(true);
    } else {
      showExPopover(false);
      if (askStatus) { askStatus.textContent = ''; askStatus.style.display = 'none'; }
    }
  } catch {}
});

function switchTab(target) {
  tabs.forEach((btn) => btn.classList.toggle('active', btn.dataset.tab === target));
  // If switching to Ask, pre-measure layout by briefly making it render offscreen
  if (target === 'ask') {
    const askPanel = document.getElementById('ask');
    if (askPanel) {
      // Remove actives first
      panels.forEach((p) => p.classList.remove('active'));
      askPanel.classList.add('premeasure');
      // Two-frame measure so CSS vars apply before we show the panel
      requestAnimationFrame(() => {
        measureLayoutVars();
        requestAnimationFrame(() => {
          measureLayoutVars();
          askPanel.classList.remove('premeasure');
          askPanel.classList.add('active');
          // Subtle child fade/slide without changing layout of container
          askPanel.classList.add('ask-appear');
          setTimeout(() => askPanel.classList.remove('ask-appear'), 320);
          // Ensure bottom observer and snap to last message top
          ensureBottomSentinel();
          initScrollObserver();
          try {
            const last = chatWindow?.querySelector('.message:last-of-type');
            if (last) {
              scrollMessageStart(last);
              requestAnimationFrame(() => scrollMessageStart(last));
            } else {
              scrollChatToBottom();
            }
            updateScrollDownBtn();
          } catch {}
        });
      });
      return;
    }
  }
  panels.forEach((panel) => panel.classList.toggle('active', panel.id === target));
  requestAnimationFrame(measureLayoutVars);
}

tabs.forEach((btn) =>
  btn.addEventListener('click', () => {
    switchTab(btn.dataset.tab);
  })
);

function setInputFiles(filesArray) {
  const input = document.getElementById('upload-input');
  if (!input || !window.DataTransfer) return;
  const dt = new DataTransfer();
  selectedFiles = Array.from(filesArray);
  selectedFiles.forEach((file) => dt.items.add(file));
  input.files = dt.files;
}

function renderChatAttachments() {
  try {
    if (!attachList) return;
    if (!chatAttachments.length) {
      attachList.innerHTML = '';
      attachList.style.display = 'none';
      return;
    }
    attachList.style.display = '';
    attachList.innerHTML = chatAttachments.map((file, idx) => {
      try { if (!file._url) file._url = URL.createObjectURL(file); } catch {}
      const url = file._url || '';
      return `
      <li style="display:inline-flex; align-items:center; gap:6px; background:rgba(15,23,42,0.06); color:#0f172a; border-radius:16px; padding:4px 8px; margin:0 6px 6px 0;">
        <img src="${url}" alt="" style="width:22px;height:22px;object-fit:cover;border-radius:6px; border:1px solid rgba(15,23,42,0.15);" />
        <span style="font-size:.9em; max-width:220px; white-space:nowrap; text-overflow:ellipsis; overflow:hidden; display:inline-block;" title="${escapeHTML(file.name)}">${escapeHTML(file.name)}</span>
        <button type="button" class="remove-attach" data-index="${idx}" title="Remove" aria-label="Remove" style="border:0; background:transparent; color:#0f172a; cursor:pointer;">×</button>
      </li>`;
    }).join('');
  } catch {}
}

attachBtn?.addEventListener('click', () => imageInput?.click());
// Fallback in case the button is re-rendered
document.addEventListener('click', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  const btn = t.closest('#attach-image');
  if (btn) {
    e.preventDefault();
    try { imageInput?.click(); } catch {}
  }
});
imageInput?.addEventListener('change', (e) => {
  try {
    const files = Array.from(imageInput.files || []).filter(f => f && f.type && f.type.startsWith('image/'));
    if (!files.length) return;
    // Limit to 4 attachments to avoid huge payloads
    const remain = Math.max(0, 4 - chatAttachments.length);
    chatAttachments.push(...files.slice(0, remain));
    renderChatAttachments();
    // reset input to allow re-selecting the same file
    imageInput.value = '';
  } catch {}
});

attachList?.addEventListener('click', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  const rm = t.closest('button.remove-attach');
  if (rm) {
    const i = Number(rm.getAttribute('data-index') || '-1');
    if (i >= 0 && i < chatAttachments.length) {
      chatAttachments.splice(i, 1);
      renderChatAttachments();
    }
  }
});

// Paste images into the question textarea
document.getElementById('question')?.addEventListener('paste', (evt) => {
  try {
    const items = evt.clipboardData && evt.clipboardData.items ? Array.from(evt.clipboardData.items) : [];
    const imgs = items.map(it => (it && it.type && it.type.startsWith('image/')) ? it.getAsFile() : null).filter(Boolean);
    if (!imgs.length) return;
    const remain = Math.max(0, 4 - chatAttachments.length);
    chatAttachments.push(...imgs.slice(0, remain));
    renderChatAttachments();
  } catch {}
});

async function filesToDataURLs(files) {
  const out = [];
  for (const f of files) {
    if (!f) continue;
    const dataUrl = await new Promise((resolve, reject) => {
      try {
        const r = new FileReader();
        r.onload = () => resolve(r.result);
        r.onerror = (e) => reject(e);
        r.readAsDataURL(f);
      } catch (e) { resolve(null); }
    });
    if (dataUrl && typeof dataUrl === 'string') {
      out.push({ name: f.name || 'image', type: f.type || 'image/png', data: dataUrl });
    }
  }
  return out;
}

// Create a small JPEG thumbnail DataURL from a larger DataURL to keep storage light
async function toThumbDataURL(dataUrl, maxW = 96, quality = 0.6) {
  return new Promise((resolve) => {
    try {
      const img = new Image();
      img.onload = () => {
        try {
          const w = img.width || 1, h = img.height || 1;
          const scale = Math.min(1, maxW / Math.max(1, w));
          const tw = Math.max(1, Math.round(w * scale));
          const th = Math.max(1, Math.round(h * scale));
          const canvas = document.createElement('canvas');
          canvas.width = tw; canvas.height = th;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, tw, th);
          const out = canvas.toDataURL('image/jpeg', quality);
          resolve(out || dataUrl);
        } catch { resolve(dataUrl); }
      };
      img.onerror = () => resolve(dataUrl);
      img.src = dataUrl;
    } catch { resolve(dataUrl); }
  });
}

async function makeThumbs(imgs, maxW = 96, quality = 0.6) {
  const thumbs = [];
  for (const it of (imgs || [])) {
    if (!it || !it.data) continue;
    const t = await toThumbDataURL(it.data, maxW, quality);
    thumbs.push({ name: it.name || 'image', type: 'image/jpeg', data: t });
  }
  return thumbs;
}

function renderFileList(fileList, opts = {}) {
  const { hideProgress = true } = opts;
  const listEl = uploadListEl;
  if (!listEl) return;
  if (!fileList || !fileList.length) {
    listEl.innerHTML = '';
    if (hideProgress) {
      if (uploadProgressBar) {
        uploadProgressBar.style.width = '0%';
        uploadProgressBar.classList.remove('active');
      }
      if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
      if (uploadProgressWrap) uploadProgressWrap.classList.remove('show');
    }
    return;
  }
  listEl.innerHTML = Array.from(fileList)
    .map(
      (file, idx) =>
        `<li class="upload-item">
          <div class="file-main">
            <div class="file-name">${escapeHTML(file.name)}</div>
            <div class="file-meta">${(file.size / 1024).toFixed(1)} KB</div>
          </div>
          <div class="file-actions">
            <button type="button" class="remove" data-remove-index="${idx}">Remove</button>
          </div>
        </li>`
    )
    .join('');

  // show progress track when files present
  if (uploadProgressTrack) uploadProgressTrack.style.display = 'block';
  if (uploadProgressWrap) uploadProgressWrap.classList.add('show');
}

function sortedFilteredLibrary(items, term, sortKey, dir='desc') {
  try {
    const q = String(term || '').trim().toLowerCase();
    let arr = Array.isArray(items) ? [...items] : [];
    if (q) {
      arr = arr.filter(d => String(d.display_name || d.name || '').toLowerCase().includes(q));
    }
    const key = String(sortKey || 'date');
    const get = (d, k) => {
      if (k === 'pages') return Number(d.pages || d.page_count || 0);
      if (k === 'chunks') return Number(d.chunks || 0);
      if (k === 'title') return String(d.display_name || d.name || '').toLowerCase();
      if (k === 'date') return Number(d.ingested_at || 0);
      return 0;
    };
    const mult = (String(dir||'desc').toLowerCase() === 'asc') ? 1 : -1;
    if (key === 'title') arr.sort((a,b) => (get(a,'title') < get(b,'title') ? -1 : 1) * mult);
    else if (key === 'pages') arr.sort((a,b) => (get(a,'pages') - get(b,'pages')) * mult);
    else if (key === 'chunks') arr.sort((a,b) => (get(a,'chunks') - get(b,'chunks')) * mult);
    else arr.sort((a,b) => (get(a,'date') - get(b,'date')) * mult);
    return arr;
  } catch { return Array.isArray(items) ? items : []; }
}

function renderLibraryBooks(items) {
  if (!libraryGrid || !libraryEmpty) return;
  if (!Array.isArray(items) || !items.length) {
    libraryGrid.innerHTML = '';
    libraryGrid.classList.add('empty');
    libraryEmpty.textContent = 'Your library is empty. Ingest a document to get started.';
    libraryEmpty.style.display = 'block';
    return;
  }
  libraryGrid.classList.remove('empty');
  libraryEmpty.style.display = 'none';
  libraryGrid.innerHTML = items
    .map((doc) => {
      const pages = doc.pages ? `${doc.pages} page${doc.pages === 1 ? '' : 's'}` : '—';
      const chunks = doc.chunks ? `${doc.chunks} chunk${doc.chunks === 1 ? '' : 's'}` : '—';
      const summary = (doc.summary || '').trim();
  const snippet = summary ? summary.split('\n').slice(0, 12).join('\n') : 'No summary available yet.';
      const added = doc.ingested_at ? new Date(doc.ingested_at * 1000).toLocaleString() : '';
      return `
        <article class="book-card" data-name="${escapeHTML(doc.name)}">
          <header>
            <h4 class="book-title">${escapeHTML(doc.display_name || doc.name)}</h4>
          </header>
          <div class="book-meta">
            <span>${pages}</span>
            <span>${chunks}</span>
            ${added ? `<span>${escapeHTML(added)}</span>` : ''}
          </div>
          <div class="book-summary">${escapeHTML(snippet).replace(/\n/g, '<br />')}</div>
          <div class="book-actions">
            <button type="button" class="library-details" data-doc="${escapeHTML(doc.name)}">View details</button>
            <button type="button" class="library-remove" data-doc="${escapeHTML(doc.name)}">Remove</button>
          </div>
        </article>`;
    })
    .join('');
  libraryGrid.scrollTop = 0;
  // Post-process to reflect any pending removal selections
  document.querySelectorAll('.book-card').forEach((el) => {
    const name = el.getAttribute('data-name');
    if (name && pendingRemovals.has(name)) {
      el.classList.add('pending-remove');
      const b = el.querySelector('.library-remove');
      if (b) b.textContent = 'Undo';
    }
  });
  updateLibraryActionButtons();
  // Attach details click handler (open modal on card click, excluding remove button)
  libraryGrid.querySelectorAll('.book-card').forEach((el) => {
    if (el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('click', (evt) => {
      const target = evt.target;
      if (target instanceof HTMLElement && (target.closest('.library-remove') || target.closest('.library-details'))) {
        const btn = target.closest('.library-details');
        if (btn) {
          const name = btn.getAttribute('data-doc') || el.getAttribute('data-name');
          if (name) openDocDetails(name);
        }
        return;
      }
      const name = el.getAttribute('data-name');
      if (name) openDocDetails(name);
    });
  });
}

async function refreshLibrary() {
  if (!libraryGrid) return;
  libraryGrid.classList.add('loading');
  if (libraryEmpty) {
    libraryEmpty.textContent = 'Loading…';
    libraryEmpty.style.display = 'block';
  }
  libraryGrid.innerHTML = '';
  try {
    const resp = await fetch(`${API_BASE}/library`, { credentials: 'include' });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    __libraryData = Array.isArray(data.documents) ? data.documents : [];
    renderLibraryBooks(sortedFilteredLibrary(__libraryData, __librarySearch, __librarySort, __libraryDir));
    // Populate the Only document selector
    if (onlyDocSelect) {
      const prev = onlyDocSelect.value;
      const docs = (__libraryData || []).map((d) => String(d.name || '').trim()).filter(Boolean);
      onlyDocSelect.innerHTML = '<option value="">All documents</option>' +
        docs.map((n) => `<option value="${escapeHTML(n)}">${escapeHTML(n)}</option>`).join('');
      if (docs.includes(prev)) onlyDocSelect.value = prev; // restore selection
    }
  } catch (err) {
    console.warn('Failed to refresh library', err);
  } finally {
    libraryGrid.classList.remove('loading');
  }
}

// Library search/sort handlers
librarySearchInput?.addEventListener('input', () => {
  __librarySearch = String(librarySearchInput.value || '').trim();
  renderLibraryBooks(sortedFilteredLibrary(__libraryData, __librarySearch, __librarySort, __libraryDir));
});
librarySortSelect?.addEventListener('change', () => {
  __librarySort = String(librarySortSelect.value || 'date');
  renderLibraryBooks(sortedFilteredLibrary(__libraryData, __librarySearch, __librarySort, __libraryDir));
});
document.getElementById('library-order')?.addEventListener('click', (e) => {
  e.preventDefault();
  __libraryDir = (__libraryDir === 'asc') ? 'desc' : 'asc';
  const ico = document.getElementById('library-order-icon');
  if (ico) ico.textContent = (__libraryDir === 'asc') ? '↑' : '↓';
  renderLibraryBooks(sortedFilteredLibrary(__libraryData, __librarySearch, __librarySort, __libraryDir));
});

async function removeLibraryDocument(name) {
  try {
    const resp = await fetch(`${API_BASE}/library/${encodeURIComponent(name)}`, {
      method: 'DELETE',
      credentials: 'include',
    });
    if (!resp.ok) {
      const text = await resp.text();
      console.warn('Failed to remove document', text);
      return;
    }
    await refreshLibrary();
  } catch (err) {
    console.warn('Failed to remove document', err);
  }
}

function clearIngestPoll(jobId) {
  const handle = ingestPolls.get(jobId);
  if (handle) {
    clearInterval(handle);
    ingestPolls.delete(jobId);
  }
}

function startIngestPoll(jobId) {
  if (!jobId) return;
  clearIngestPoll(jobId);

  const poll = async () => {
    try {
      const resp = await fetch(`${API_BASE}/ingest/status/${jobId}`, { credentials: 'include' });
      if (!resp.ok) {
        ingestLog.textContent = `Error: ${await resp.text()}`;
        scrollIngestToBottom();
        clearIngestPoll(jobId);
        if (uploadProgressBar) uploadProgressBar.classList.remove('active');
        if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
        if (uploadProgressWrap) uploadProgressWrap.classList.remove('show');
        return;
      }
      const data = await resp.json();
      if (Array.isArray(data.logs)) {
        ingestLog.textContent = data.logs.join('\n');
        scrollIngestToBottom();
      }
      // Compute live parsing progress from page counts across all docs
      const docs = Array.isArray(data.documents) ? data.documents : [];
      let pagesDone = 0;
      let pagesTotal = 0;
      for (const d of docs) {
        const t = Number(d.pages_total || 0);
        const dn = Number(d.pages_done != null ? d.pages_done : d.pages || 0);
        if (t > 0) pagesTotal += t;
        if (dn > 0) pagesDone += Math.min(dn, t || dn);
      }
      const pctReported = Number(data.progress ?? 0);
      const pctParsing = pagesTotal > 0 ? Math.min(50, (pagesDone / Math.max(1, pagesTotal)) * 50) : 0;
      // This is the percentage we display to the user. The bar mirrors this exactly.
      const pctDisplay = pctReported <= 50 ? Math.max(pctReported, pctParsing) : pctReported;
      if (uploadProgressBar && pctDisplay >= 0) {
        uploadProgressBar.style.width = `${Math.max(0, Math.min(100, pctDisplay))}%`;
        uploadProgressBar.classList.add('active');
      }
      if (uploadProgressTrack) {
        uploadProgressTrack.style.display = 'block';
        if (uploadProgressWrap) uploadProgressWrap.classList.add('show');
      }
      const statusEl = document.getElementById('upload-progress-status');
      if (statusEl) {
        if (pagesTotal > 0 && pctReported <= 50) {
          statusEl.textContent = `Parsing… ${Math.round(pctDisplay)}%  (${pagesDone} / ${pagesTotal} pages)`;
        } else if (pctReported > 50 && pctReported < 100) {
          statusEl.textContent = `Indexing and embedding… ${Math.round(pctDisplay)}%`;
        } else if (pctReported >= 100) {
          statusEl.textContent = `Done`;
        } else {
          statusEl.textContent = '';
        }
      }
      if (data.status === 'done') {
        if (data.summary_html && summaryContent) summaryContent.innerHTML = data.summary_html;
        if (data.details_html && snapshotContent) snapshotContent.innerHTML = data.details_html;
        // Create a new chat and place the summary as the first bot message
        const firstDoc = (Array.isArray(data.documents) && data.documents.length) ? data.documents[0] : null;
        const convTitle = firstDoc?.name ? `Summary: ${firstDoc.name}` : 'Ingestion summary';
        createConversation(convTitle);
        const summaryHtml = data.summary_html || '<p>Ingestion complete.</p>';
        appendBotMessage(summaryHtml, { animate: false });
        recordConversationEntry('', summaryHtml, null, null);
        // Ask server to generate a concise title for this summary
        maybeUpdateTitleFrom(stripHTML(summaryHtml));
        ingestLog.textContent += '\nIngestion complete.';
        scrollIngestToBottom();
        clearIngestPoll(jobId);
        if (uploadProgressBar) uploadProgressBar.style.width = '100%';
        setTimeout(() => {
          if (uploadProgressBar) uploadProgressBar.classList.remove('active');
          if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
        }, 600);
        switchTab('ask');
        refreshLibrary();
        return;
      }
      if (data.status === 'error' || data.status === 'cancelled') {
        if (data.error) ingestLog.textContent += `\n${data.error}`;
        if (data.error) showError(data.error);
        scrollIngestToBottom();
        clearIngestPoll(jobId);
        if (uploadProgressBar) uploadProgressBar.classList.remove('active');
        if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
        if (uploadProgressWrap) uploadProgressWrap.classList.remove('show');
      }
    } catch (err) {
      ingestLog.textContent = `Error: ${err}`;
      scrollIngestToBottom();
      clearIngestPoll(jobId);
      if (uploadProgressBar) uploadProgressBar.classList.remove('active');
      if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
      if (uploadProgressWrap) uploadProgressWrap.classList.remove('show');
    }
  };

  poll();
  const handle = setInterval(poll, 1000);
  ingestPolls.set(jobId, handle);
}

async function uploadDocuments(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append('files', file);
  }
  ingestLog.textContent = 'Uploading...';
  scrollIngestToBottom();
  const ingestButton = uploadForm?.querySelector('.primary');
  if (ingestButton) ingestButton.disabled = true;
  if (uploadProgressBar) {
    uploadProgressBar.style.width = '35%';
    uploadProgressBar.classList.add('active');
  }
  if (uploadProgressTrack) uploadProgressTrack.style.display = 'block';
  if (uploadProgressWrap) uploadProgressWrap.classList.add('show');
  document.querySelector('#library details')?.setAttribute('open', 'open');
  const resp = await fetch(`${API_BASE}/ingest`, {
    method: 'POST',
    body: formData,
    credentials: 'include',
  });
  if (!resp.ok) {
    const text = await resp.text();
    ingestLog.textContent = `Error: ${text}`;
    scrollIngestToBottom();
    if (ingestButton) ingestButton.disabled = false;
    if (uploadProgressBar) uploadProgressBar.classList.remove('active');
    if (uploadProgressTrack) uploadProgressTrack.style.display = 'none';
    if (uploadProgressWrap) uploadProgressWrap.classList.remove('show');
    return;
  }
  const data = await resp.json();
  if (Array.isArray(data.logs)) {
    ingestLog.textContent = data.logs.join('\n');
  } else if (data.log) {
    ingestLog.textContent = data.log;
  }
  scrollIngestToBottom();
  const pct = Number(data.progress ?? 0);
  if (uploadProgressBar && pct > 0) {
    uploadProgressBar.style.width = `${pct}%`;
    uploadProgressBar.classList.add('active');
  }
  if (uploadProgressTrack) uploadProgressTrack.style.display = 'block';
  if (uploadProgressWrap) uploadProgressWrap.classList.add('show');
  if (data.job_id) {
    startIngestPoll(data.job_id);
  }
  const input = document.getElementById('upload-input');
  if (input) {
    input.value = '';
    renderFileList([], { hideProgress: false });
  }
  const ingestButtonAfter = uploadForm?.querySelector('.primary');
  if (ingestButtonAfter) ingestButtonAfter.disabled = false;
}

uploadForm?.addEventListener('submit', (evt) => {
  evt.preventDefault();
  if (!selectedFiles.length) {
    ingestLog.textContent = 'Select one or more documents with text.';
    scrollIngestToBottom();
    return;
  }
  renderFileList(selectedFiles);
  uploadDocuments(selectedFiles);
});

const uploadInput = document.getElementById('upload-input');
uploadInput?.addEventListener('change', () => {
  const added = Array.from(uploadInput.files || []);
  // Append to persistent selection and de-duplicate by name+size
  const map = new Map(selectedFiles.map(f => [f.name + ':' + f.size, f]));
  for (const f of added) map.set(f.name + ':' + f.size, f);
  selectedFiles = Array.from(map.values());
  setInputFiles(selectedFiles);
  renderFileList(selectedFiles);
});

uploadListEl?.addEventListener('click', (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLElement)) return;
  const attr = target.getAttribute('data-remove-index');
  if (attr === null) return;
  const idx = Number(attr);
  if (Number.isNaN(idx)) return;
  selectedFiles.splice(idx, 1);
  setInputFiles(selectedFiles);
  renderFileList(selectedFiles);
});

// ---- Drag and drop support for upload zone ----
if (dropZone) {
  ['dragenter','dragover'].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
      if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
    });
  });
  ['dragleave','dragend','drop'].forEach((ev) => {
    dropZone.addEventListener(ev, (e) => {
      if (ev !== 'drop') {
        dropZone.classList.remove('dragover');
      }
    });
  });
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer?.files || []);
    if (!files.length) return;
    const map = new Map(selectedFiles.map(f => [f.name + ':' + f.size, f]));
    for (const f of files) map.set(f.name + ':' + f.size, f);
    selectedFiles = Array.from(map.values());
    setInputFiles(selectedFiles);
    renderFileList(selectedFiles);
  });
}

// Prevent the browser from navigating when dropping outside the zone
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => {
  if (!(e.target instanceof HTMLElement) || !e.target.closest('.ingest-dropzone')) {
    e.preventDefault();
  }
});

// ---- Chat helpers ----
if (window.marked && typeof window.marked.setOptions === 'function') {
  window.marked.setOptions({ gfm: true, breaks: true });
}

function updateScrollDownBtn() {
  try {
    if (!chatBody || !scrollDownBtn) return;
    const maxScroll = Math.max(0, chatBody.scrollHeight - chatBody.clientHeight);
    const scrollable = maxScroll > 1;
    // Math check always available; sentinel refines it
    const nearBottom = chatBody.scrollTop >= (maxScroll - 4);
    const atBottom = hasSentinelSignal ? (nearBottom && isAtBottom) : nearBottom;
    scrollDownBtn.classList.toggle('show', scrollable && !atBottom);
  } catch {}
}

function scrollMessageStart(el) {
  try {
    if (!chatBody || !el) return;
    // Compute desired top alignment using rects to avoid offsetParent issues
    const rootRect = chatBody.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    const delta = elRect.top - rootRect.top;
    const target = Math.min(
      Math.max(0, chatBody.scrollTop + delta),
      Math.max(0, chatBody.scrollHeight - chatBody.clientHeight)
    );
    chatBody.scrollTop = target;
    // Fallback for browsers/layouts where manual math doesn't settle
    try { el.scrollIntoView({ block: 'start', inline: 'nearest' }); } catch {}
  } catch {}
}

function ensureTempSpacer(heightPx = 0) {
  if (!chatWindow) return null;
  if (!tempBottomSpacer || !tempBottomSpacer.isConnected) {
    tempBottomSpacer = document.createElement('div');
    tempBottomSpacer.id = 'temp-bottom-spacer';
    tempBottomSpacer.style.cssText = 'width:100%;height:0px;pointer-events:none;';
    chatWindow.appendChild(tempBottomSpacer);
  }
  const h = Math.max(0, Math.floor(heightPx));
  tempBottomSpacer.style.height = h + 'px';
  // Keep sentinel as the last element so the observer remains correct
  try { ensureBottomSentinel(); } catch {}
  return tempBottomSpacer;
}

function removeTempSpacer() {
  try {
    if (tempBottomSpacer && tempBottomSpacer.isConnected) tempBottomSpacer.remove();
  } catch {}
  tempBottomSpacer = null;
  pinnedTopMessage = null;
}

function computeNeededSpacerForTop(el) {
  try {
    if (!chatBody || !el) return 0;
    const rootRect = chatBody.getBoundingClientRect();
    const elRect = el.getBoundingClientRect();
    const delta = elRect.top - rootRect.top; // desired scroll delta to top
    const desired = chatBody.scrollTop + delta;
    const maxScroll = Math.max(0, chatBody.scrollHeight - chatBody.clientHeight);
    const needed = Math.max(0, Math.ceil(desired - maxScroll));
    return needed;
  } catch { return 0; }
}

const TEMP_SPACER_VH_FRAC = 0.375; // ~37.5% of viewport height
function getTempSpacerCap() {
  try {
    const h = Math.max(0, window.innerHeight || document.documentElement?.clientHeight || chatBody?.clientHeight || 0);
    return Math.max(0, Math.round(h * TEMP_SPACER_VH_FRAC));
  } catch { return 0; }
}
function alignMessageTopWithSpacer(el) {
  try {
    const needed = computeNeededSpacerForTop(el);
    if (needed > 0) {
      ensureTempSpacer(Math.min(needed, getTempSpacerCap()));
      pinnedTopMessage = el;
    } else {
      removeTempSpacer();
    }
    scrollMessageStart(el);
  } catch {}
}

function scrollChatToBottom() {
  if (!chatWindow) return;
  ensureBottomSentinel();
  if (bottomSentinel && typeof bottomSentinel.scrollIntoView === 'function') {
    try { bottomSentinel.scrollIntoView({ block: 'end' }); } catch {}
  }
  if (chatBody) {
    const maxScroll = Math.max(0, chatBody.scrollHeight - chatBody.clientHeight);
    chatBody.scrollTop = maxScroll;
  } else {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }
  updateScrollDownBtn();
}

function initScrollObserver() {
  if (!chatBody || !chatWindow) return;
  ensureBottomSentinel();
  try { if (sentinelObserver) sentinelObserver.disconnect(); } catch {}
  try {
    hasSentinelSignal = false;
    sentinelObserver = new IntersectionObserver((entries) => {
      const entry = entries && entries[0];
      const ratio = entry ? entry.intersectionRatio : 0;
      const vis = Boolean(entry && entry.isIntersecting && ratio >= 0.999);
      hasSentinelSignal = true;
      isAtBottom = vis;
      updateScrollDownBtn();
    }, { root: chatBody, threshold: 1.0 });
    if (bottomSentinel) sentinelObserver.observe(bottomSentinel);
  } catch {}
}

function typesetMath() {
  if (!chatWindow) return Promise.resolve();
  if (window.MathJax && typeof window.MathJax.typesetPromise === 'function') {
    return window.MathJax.typesetPromise([chatWindow]).catch(() => {});
  }
  return Promise.resolve();
}

function createMessageElement(type, content, opts = {}) {
  if (!chatWindow) return null;
  hideEmptyChatTitle();
  const animate = opts.animate !== false;
  const el = document.createElement('div');
  el.className = `message ${type}` + (animate ? ' entering' : '');
  if (opts.html) {
    el.innerHTML = content;
  } else {
    el.textContent = content;
  }
  chatWindow.appendChild(el);
  if (animate) {
    el.addEventListener('animationend', () => el.classList.remove('entering'), { once: true });
  }
  return el;
}

function appendUserMessage(text, options = {}) {
  const el = createMessageElement('user', text, { animate: options.animate !== false, html: options && options.html === true });
  if (sidebarMedia.matches) setSidebarOpen(false);
  ensureBottomSentinel();
  if (options.scroll !== false) {
    alignMessageTopWithSpacer(el);
    requestAnimationFrame(() => alignMessageTopWithSpacer(el));
    setTimeout(() => alignMessageTopWithSpacer(el), 120);
    setTimeout(() => alignMessageTopWithSpacer(el), 240);
  } else {
    updateScrollDownBtn();
  }
  // Mark this conversation as recently interacted with (skip when replaying history)
  try {
    const isReplay = options && options.scroll === false && options.animate === false;
    if (!isReplay && activeConvId && conversations[activeConvId]) {
      const conv = conversations[activeConvId];
      conv.updatedAt = Date.now();
      conversations[activeConvId] = conv;
      saveConversations();
      renderConversationList();
    }
  } catch {}
  return el;
}

function appendBotMessage(html, options = {}) {
  const el = createMessageElement('bot', html, { animate: options.animate !== false, html: true });
  const shouldScroll = options.scroll !== false;
  ensureBottomSentinel();
  // Remove any temporary padding added for aligning user message
  removeTempSpacer();
  if (shouldScroll) {
    scrollMessageStart(el);
    requestAnimationFrame(() => scrollMessageStart(el));
    setTimeout(() => scrollMessageStart(el), 120);
    setTimeout(() => scrollMessageStart(el), 240);
  } else {
    updateScrollDownBtn();
  }
  if (options.typeset !== false) {
    typesetMath().then(() => {
      removeTempSpacer();
      if (shouldScroll) {
        scrollMessageStart(el);
        requestAnimationFrame(() => scrollMessageStart(el));
        setTimeout(() => scrollMessageStart(el), 120);
      }
    });
  }
  return el;
}

function renderTraceHTML(trace) {
  try {
    if (!trace) return '';
    const retr = Array.isArray(trace.retrieval) ? trace.retrieval : [];
    const ctx = Array.isArray(trace.selected_context) ? trace.selected_context : [];
    const web = Array.isArray(trace.web_sources) ? trace.web_sources : [];
    const quotes = Array.isArray(trace.quotes) ? trace.quotes : [];
    const citations = Array.isArray(trace.citations) ? trace.citations : [];
    const reranker = String(trace.reranker || 'off');
    const rerankApplied = Boolean(trace.rerank_applied);
    const retrievalHead = Array.isArray(trace.retrieval_head) ? trace.retrieval_head : [];
    const selectedHead = Array.isArray(trace.selected_head) ? trace.selected_head : [];
    const esc = escapeHTML;
    let html = '<details class="trace"><summary>Trace</summary>';
    // Reranker info
    html += `<div><strong>Reranker:</strong> ${esc(reranker)}${rerankApplied ? ' (applied)' : ' (off)'}</div>`;
    if (retrievalHead.length || selectedHead.length) {
      html += '<div style="margin-top:6px"><strong>Order (first 5)</strong><div style="display:flex; gap:12px; flex-wrap:wrap">'
        + `<div><em>Retrieval</em><ul>${retrievalHead.map(x => `<li>${esc(String(x))}</li>`).join('')}</ul></div>`
        + `<div><em>Selected</em><ul>${selectedHead.map(x => `<li>${esc(String(x))}</li>`).join('')}</ul></div>`
        + '</div></div>';
    }
    const moves = Array.isArray(trace.rerank_moves) ? trace.rerank_moves : [];
    if (moves.length) {
      html += '<div style="margin-top:6px"><strong>Reorder impact</strong><ul>'
        + moves.slice(0, 10).map(m => `<li>${esc(String(m.key))} — moved ${m.delta > 0 ? 'up' : 'down'} ${Math.abs(m.delta)} (old ${m.old+1} → new ${m.new+1})</li>`).join('')
        + '</ul></div>';
    }
    const citeCheck = Array.isArray(trace.citation_check) ? trace.citation_check : [];
    if (citeCheck.length) {
      const bad = citeCheck.filter(c => !c.ok);
      const good = citeCheck.filter(c => c.ok);
      html += '<div style="margin-top:6px"><strong>Citation validation</strong>'; 
      if (bad.length) {
        html += '<div style="color:#991b1b">Weak/Unmatched: ' + bad.map(c => esc(String(c.tag))).join(', ') + '</div>';
      }
      if (good.length) {
        html += '<div>OK: ' + good.map(c => esc(String(c.tag))).join(', ') + '</div>';
      }
      html += '</div>';
    }
    if (retr.length) {
      html += '<div><strong>Top retrieval</strong><ul>' + retr.map(r => {
        const tag = `[${esc(String(r.doc || '?'))} p.${esc(String(r.page || '?'))}]`;
        const sc = (typeof r.score === 'number') ? r.score.toFixed(3) : esc(String(r.score || ''));
        return `<li>${tag} — score ${sc}</li>`;
      }).join('') + '</ul></div>';
    }
    if (ctx.length) {
      html += '<div style="margin-top:6px"><strong>Selected context</strong>' + ctx.map(c => {
        const tag = `[${esc(String(c.doc || '?'))} p.${esc(String(c.page || '?'))}]`;
        const sc = (typeof c.score === 'number') ? c.score.toFixed(3) : esc(String(c.score || ''));
        const sn = esc(String(c.snippet || '')).slice(0, 400);
        const url = c.url ? ` <a href="${esc(c.url)}" target="_blank" rel="noreferrer">link</a>` : '';
        const view = (!c.url && String(c.kind || 'doc') === 'doc') ? ` <a href="#" class="view-chip" data-view-doc="${esc(String(c.doc||''))}" data-view-page="${esc(String(c.page||''))}" data-view-needle="${esc(String(c.needle||''))}">view</a>` : '';
        return `<div style="margin:4px 0">${tag} — ${esc(String(c.kind || 'doc'))} — score ${sc}${url}${view}<br/><small>${sn}</small></div>`;
      }).join('') + '</div>';
    }
    if (citations.length) {
      html += '<div style="margin-top:6px"><strong>Citations</strong><div>' + citations.map(c => `<span style="display:inline-block; margin:2px 6px 2px 0; padding:2px 6px; border-radius:10px; background:rgba(15,23,42,0.08)">${escapeHTML(String(c))}</span>`).join('') + '</div></div>';
    }
    if (web.length) {
      html += '<div style="margin-top:6px"><strong>Web sources</strong><ul>' + web.map(u => `<li><a href="${esc(String(u))}" target="_blank" rel="noreferrer">${esc(String(u))}</a></li>`).join('') + '</ul></div>';
    }
    if (quotes.length) {
      html += '<div style="margin-top:6px"><strong>Evidence snippets</strong>' + quotes.map(q => {
        const text = esc(String(q.quote || q.text || ''));
        const src = esc(String(q.source || q.citation || ''));
        return `<blockquote style="margin:6px 0; padding-left:10px; border-left:3px solid rgba(15,23,42,0.2)">${text}${src ? `<div><small>${src}</small></div>` : ''}</blockquote>`;
      }).join('') + '</div>';
    }
    html += '</details>';
    return html;
  } catch (e) {
    try { console.warn('trace render failed', e); } catch {}
    return '';
  }
}

// Inline doc page viewer
function openPageViewer(doc, page, needle) {
  try {
    if (!pageViewer || !pageViewerImg) return;
    resetViewerTransform();
    pageViewer.style.display = 'flex';
    pageViewerImg.src = '';
    const base = API_BASE.replace(/\/api$/, '');
    const url = `${base}/api/library/page_image?doc=${encodeURIComponent(doc)}&page=${encodeURIComponent(page)}${needle ? `&needle=${encodeURIComponent(needle)}` : ''}`;
    fetch(url, { credentials: 'include' })
      .then(r => r.blob())
      .then(b => { pageViewerImg.src = URL.createObjectURL(b); })
      .catch(() => { try { pageViewer.style.display = 'none'; } catch {} });
  } catch {}
}
pageViewer?.addEventListener('click', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.classList.contains('doc-backdrop')) {
    pageViewer.style.display = 'none';
    try { pageViewerImg.src = ''; } catch {}
  }
});

// Generic image preview using the same viewer
function openImagePreview(url) {
  try {
    if (!pageViewer || !pageViewerImg) return;
    resetViewerTransform();
    pageViewer.style.display = 'flex';
    pageViewerImg.src = url || '';
    // Ensure transform applies after the image has natural dimensions
    try {
      pageViewerImg.onload = () => { resetViewerTransform(); };
    } catch {}
  } catch {}
}

document.addEventListener('click', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  const link = t.closest('[data-view-doc]');
  if (link) {
    e.preventDefault();
    const d = link.getAttribute('data-view-doc');
    const p = link.getAttribute('data-view-page');
    const needle = link.getAttribute('data-view-needle') || '';
    if (d && p) openPageViewer(d, p, needle);
    return;
  }
  const userImg = t.closest('.user-attach-grid img');
  if (userImg && userImg instanceof HTMLImageElement) {
    e.preventDefault();
    const full = userImg.getAttribute('data-full') || userImg.getAttribute('src');
    if (full) {
      try { openImagePreview(full); } catch {}
    }
    return;
  }
  const rc = t.closest('button.retry-chip');
  if (rc) {
    e.preventDefault();
    const q = document.getElementById('question')?.value || '';
    const payload = collectAskPayloadFromUI();
    if (rc.hasAttribute('data-retry-web')) payload.web_enabled = true;
    if (rc.hasAttribute('data-retry-broaden')) delete payload.only_doc;
    askQuestion(payload, q || (chatWindow?.querySelector('.message.user:last-of-type')?.textContent || ''));
    return;
  }
});

function collectAskPayloadFromUI() {
  return {
    question: (document.getElementById('question')?.value || '').trim(),
    memory_enabled: true,
    formula_mode: document.getElementById('formula-mode').checked,
    strict_docs: document.getElementById('strict-docs')?.checked || false,
    web_enabled: document.getElementById('web-enabled').checked,
    exhaustive: document.getElementById('exhaustive').checked,
    reranker: (() => {
      const on = document.getElementById('reranker-on')?.checked;
      if (!on) return 'off';
      const chosen = (defaultsData && typeof defaultsData.ASK_RERANKER === 'string' && defaultsData.ASK_RERANKER.toLowerCase() !== 'off')
        ? String(defaultsData.ASK_RERANKER)
        : 'minilm';
      return chosen;
    })(),
    top_k: Number(document.getElementById('top-k').value || 10),
    max_batches: Number(document.getElementById('max-batches').value || 6),
    time_budget: Number(document.getElementById('time-budget').value || 120),
    history: chatHistory,
    ...(onlyDocSelect && onlyDocSelect.value ? { only_doc: onlyDocSelect.value } : {}),
  };
}

function maybeAppendTrace(trace) {
  try {
    const html = renderTraceHTML(trace);
    if (!html) return;
    // Ensure only the latest message has a trace chip
    try {
      chatWindow?.querySelectorAll('.trace-inline, details.trace').forEach((el) => el.remove());
    } catch {}
    const lastBot = chatWindow?.querySelector('.message.bot:last-of-type');
    if (lastBot) {
      const wrap = document.createElement('div');
      wrap.className = 'trace-inline';
      wrap.innerHTML = html;
      lastBot.appendChild(wrap);
    } else {
      // Fallback: insert as a separate message
      appendBotMessage(html, { typeset: false, animate: false, scroll: false });
    }
  } catch {}
}

function maybeInsertSummaryMessage(summaryHtml) {
  const hasAny = chatHistory.length > 0 || chatWindow.innerHTML.trim().length > 0;
  if (!hasAny && summaryHtml) {
    appendBotMessage(summaryHtml);
  }
}

let askPollHandle = null;
function updateAskStatusFromLogs(logs) {
  try {
    if (!askStatus) return;
    const exOn = !!document.getElementById('exhaustive')?.checked;
    const arr = Array.isArray(logs) ? logs : [];
    let status = '';
    // Priority: Batch progress > Sweep plan > Reranker > MMR/Searching
    for (let i = arr.length - 1; i >= 0; i--) {
      const ln = String(arr[i] || '');
      if (/^Batch\s+\d+\/\d+/.test(ln)) { status = ln; break; }
    }
    if (!status) {
      for (let i = arr.length - 1; i >= 0; i--) {
        const ln = String(arr[i] || '');
        if (ln.startsWith('Sweep plan:')) { status = ln; break; }
      }
    }
    if (!status) {
      for (let i = arr.length - 1; i >= 0; i--) {
        const ln = String(arr[i] || '');
        if (ln.startsWith('Reranker:')) { status = ln; break; }
      }
    }
    if (!status) {
      for (let i = arr.length - 1; i >= 0; i--) {
        const ln = String(arr[i] || '');
        if (/^(MMR selection|MMR done|Searching index|Summarizing with context)/.test(ln)) { status = ln; break; }
      }
    }
    askStatus.textContent = status || '';
    askStatus.style.display = (exOn && status) ? 'block' : 'none';
  } catch {}
}
async function askQuestion(payload, question) {
  if (askLog) askLog.textContent = 'Working…';
  const sendBtn = askForm?.querySelector('.send-btn');
  if (sendBtn) sendBtn.disabled = true;
  if (askProgress) askProgress.classList.add('active');
  try { showExPopover(false); } catch {}
  let resp;
  try {
    resp = await fetch(`${API_BASE}/ask/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      credentials: 'include',
    });
  } catch (err) {
    if (askLog) askLog.textContent = `Error: ${err}`;
    if (sendBtn) sendBtn.disabled = false;
    if (askProgress) askProgress.classList.remove('active');
    return;
  }
  if (!resp.ok) {
    const text = await resp.text();
    // Fallback to synchronous /api/ask if /api/ask/start is unavailable
    if (resp.status === 404) {
      try {
        const direct = await fetch(`${API_BASE}/ask`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
          credentials: 'include',
        });
        if (!direct.ok) {
          const t = await direct.text();
          if (askLog) askLog.textContent = `Error: ${t}`;
          showError(t || 'Ask failed');
          if (sendBtn) sendBtn.disabled = false;
          return;
        }
        const d = await direct.json();
        if (askLog) { askLog.textContent = d.log || 'Ready.'; try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
        if (d.answer) {
          appendBotMessage(d.answer);
        }
        if (d.trace) { maybeAppendTrace(d.trace); }
        if (question && !cancelArmed) {
          recordConversationEntry(question, d.answer || '', d.answer_markdown, pendingUserHtml || null);
          pendingUserHtml = null;
        }
        if (d.answer) {
          const plain = stripHTML(d.answer);
          maybeUpdateTitleFrom(question + ' \n' + plain);
        }
      } catch (e) {
        if (askLog) askLog.textContent = `Error: ${e}`;
        showError(e);
      }
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    } else {
      if (askLog) askLog.textContent = `Error: ${text}`;
      showError(text || 'Ask failed');
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    }
  }
  const data = await resp.json();
  const jobId = data.job_id;
  if (!jobId) {
    if (askLog) askLog.textContent = 'Error: missing job id';
    if (sendBtn) sendBtn.disabled = false;
    if (askProgress) askProgress.classList.remove('active');
    return;
  }
  if (askPollHandle) { clearInterval(askPollHandle); askPollHandle = null; }
  try { localStorage.setItem('anag_active_job', jobId); } catch {}
  // Turn the send button into a Cancel button during the run
  let cancelArmed = false;
  const sendButtonEl = askForm?.querySelector('.send-btn');
  if (sendButtonEl) {
    sendButtonEl.disabled = false;
    sendButtonEl.title = 'Cancel';
    sendButtonEl.classList.add('is-cancel');
    try { sendButtonEl.setAttribute('type', 'button'); } catch {}
    const cancelOnce = async (e) => {
      if (cancelArmed) return; cancelArmed = true;
      e?.preventDefault?.();
      try { await fetch(`${API_BASE}/ask/cancel/${jobId}`, { method: 'POST', credentials: 'include' }); } catch {}
      // Move cancellation notice to Ask Log; stop progress; hide inline status
      try {
        if (askLog) {
          const prev = String(askLog.textContent || '').trim();
          askLog.textContent = prev ? (prev + '\nCancelled') : 'Cancelled';
          try { askLog.scrollTop = askLog.scrollHeight; } catch {}
        }
      } catch {}
      try { if (askProgress) askProgress.classList.remove('active'); } catch {}
      try { if (askStatus) { askStatus.textContent = ''; askStatus.style.display = 'none'; } } catch {}
      try {
        // Persist the user's question as a turn with an empty answer so it remains in conversation memory
        if (question) {
          recordConversationEntry(question, '', '', pendingUserHtml || null);
          pendingUserHtml = null;
        }
      } catch {}
      try { if (sendButtonEl) { sendButtonEl.classList.remove('is-cancel'); sendButtonEl.title = 'Send'; sendButtonEl.setAttribute('type', 'submit'); } } catch {}
      try { if (askPollHandle) { clearInterval(askPollHandle); askPollHandle = null; } } catch {}
      try { localStorage.removeItem('anag_active_job'); } catch {}
      try { removeTempSpacer(); } catch {}
      try { updateScrollDownBtn(); } catch {}
    };
    try { sendButtonEl.addEventListener('click', cancelOnce, { once: true }); } catch {}
  }
  const poll = async () => {
    try {
      const s = await fetch(`${API_BASE}/ask/status/${jobId}`, { credentials: 'include' });
      if (!s.ok) {
        if (s.status === 404) {
          // Fallback: server missing status route; perform direct ask once.
          clearInterval(askPollHandle); askPollHandle = null;
          try {
            const direct = await fetch(`${API_BASE}/ask`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
              credentials: 'include',
            });
            if (!direct.ok) {
              const t = await direct.text();
              if (askLog) askLog.textContent = `Error: ${t}`;
              showError(t || 'Ask failed');
            } else {
              const d = await direct.json();
              if (askLog) { askLog.textContent = d.log || 'Ready.'; try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
              if (d.answer) {
                appendBotMessage(d.answer);
              }
              if (question) {
                recordConversationEntry(question, d.answer || '', d.answer_markdown, pendingUserHtml || null);
                pendingUserHtml = null;
              }
              if (d.answer) {
                const plain = stripHTML(d.answer);
                maybeUpdateTitleFrom(question + ' \n' + plain);
              }
            }
          } catch (e) {
            if (askLog) askLog.textContent = `Error: ${e}`;
            showError(e);
          }
          if (sendBtn) sendBtn.disabled = false;
          if (askProgress) askProgress.classList.remove('active');
          return;
        } else {
          const txt = await s.text();
          if (askLog) askLog.textContent = `Error: ${txt}`;
          showError(txt || 'Ask status failed');
          clearInterval(askPollHandle); askPollHandle = null;
          if (sendBtn) sendBtn.disabled = false;
          if (askProgress) askProgress.classList.remove('active');
          return;
        }
      }
      const st = await s.json();
      if (Array.isArray(st.logs)) updateAskStatusFromLogs(st.logs);
      if (Array.isArray(st.logs) && askLog) { askLog.textContent = st.logs.join('\n'); try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
      if (st.status === 'done') {
        clearInterval(askPollHandle); askPollHandle = null;
        try { updateAskStatusFromLogs([]); } catch {}
        if (st.answer) {
          appendBotMessage(st.answer);
        }
        if (st.trace) { maybeAppendTrace(st.trace); }
        if (question && !cancelArmed) {
          recordConversationEntry(question, st.answer || '', st.answer_markdown, pendingUserHtml || null);
          pendingUserHtml = null;
        }
        if (st.answer) {
          const plain = stripHTML(st.answer);
          maybeUpdateTitleFrom(question + ' \n' + plain);
        }
        if (sendBtn) sendBtn.disabled = false;
        if (askProgress) askProgress.classList.remove('active');
        try { localStorage.removeItem('anag_active_job'); } catch {}
        try { if (sendButtonEl) { sendButtonEl.classList.remove('is-cancel'); sendButtonEl.title = 'Send'; sendButtonEl.setAttribute('type','submit'); } } catch {}
        // Inline retry chips if server suggests
        try {
          const meta = st.meta || {};
          if (meta && (meta.not_found || meta.suggest_web || meta.suggest_broaden)) {
            const last = chatWindow?.querySelector('.message.bot:last-of-type');
            if (last) {
              const wrap = document.createElement('div');
              wrap.style.marginTop = '6px';
              wrap.className = 'retry-wrap';
              const chips = [];
              if (meta.suggest_web) chips.push('<button class="retry-chip" data-retry-web="1">Retry with Web</button>');
              if (meta.suggest_broaden) chips.push('<button class="retry-chip" data-retry-broaden="1">Broaden to all docs</button>');
              wrap.innerHTML = `<div class="retry-chips">${chips.join(' ')}</div>`;
              last.appendChild(wrap);
            }
          }
        } catch {}
      } else if (st.status === 'cancelled') {
        clearInterval(askPollHandle); askPollHandle = null;
        try { updateAskStatusFromLogs([]); } catch {}
        // Log cancellation to Ask Log, no error banner
        if (askLog) {
          const prev = String(askLog.textContent || '').trim();
          askLog.textContent = prev ? (prev + '\nCancelled') : 'Cancelled';
          try { askLog.scrollTop = askLog.scrollHeight; } catch {}
        }
        if (sendBtn) sendBtn.disabled = false;
        if (askProgress) askProgress.classList.remove('active');
        try { localStorage.removeItem('anag_active_job'); } catch {}
        try { if (sendButtonEl) { sendButtonEl.classList.remove('is-cancel'); sendButtonEl.title = 'Send'; sendButtonEl.setAttribute('type','submit'); } } catch {}
        return;
      } else if (st.status === 'error') {
        clearInterval(askPollHandle); askPollHandle = null;
        try { updateAskStatusFromLogs([]); } catch {}
        if (askLog) askLog.textContent += `\n${st.error || st.status}`;
        showError(st.error || st.status || 'Ask failed');
        if (sendBtn) sendBtn.disabled = false;
        if (askProgress) askProgress.classList.remove('active');
        try { localStorage.removeItem('anag_active_job'); } catch {}
        try { if (sendButtonEl) { sendButtonEl.classList.remove('is-cancel'); sendButtonEl.title = 'Send'; sendButtonEl.setAttribute('type','submit'); } } catch {}
      }
    } catch (err) {
      if (askLog) askLog.textContent = `Error: ${err}`;
      showError(err);
      clearInterval(askPollHandle); askPollHandle = null;
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
    }
  };
  poll();
  askPollHandle = setInterval(poll, 800);
}

// Resume polling an active job if the page was refreshed
try {
  const jid = localStorage.getItem('anag_active_job');
  if (jid) {
    const resumePoll = async () => {
      try {
        const s = await fetch(`${API_BASE}/ask/status/${jid}`, { credentials: 'include' });
        if (!s.ok) { localStorage.removeItem('anag_active_job'); return; }
        const st = await s.json();
        if (Array.isArray(st.logs)) updateAskStatusFromLogs(st.logs);
        if (!st.status || st.status === 'done' || st.status === 'error' || st.status === 'cancelled') {
          localStorage.removeItem('anag_active_job');
          return;
        }
        askPollHandle = setInterval(async () => {
          const s2 = await fetch(`${API_BASE}/ask/status/${jid}`, { credentials: 'include' });
          if (!s2.ok) { clearInterval(askPollHandle); askPollHandle = null; localStorage.removeItem('anag_active_job'); return; }
          const st2 = await s2.json();
          if (Array.isArray(st2.logs)) updateAskStatusFromLogs(st2.logs);
          if (st2.status === 'done' || st2.status === 'error' || st2.status === 'cancelled') {
            clearInterval(askPollHandle); askPollHandle = null; localStorage.removeItem('anag_active_job');
            try { if (askStatus) { askStatus.textContent=''; askStatus.style.display='none'; } } catch {}
          }
        }, 900);
      } catch { localStorage.removeItem('anag_active_job'); }
    };
    setTimeout(resumePoll, 300);
  }
} catch {}

askForm?.addEventListener('submit', async (evt) => {
  evt.preventDefault();
  const question = document.getElementById('question').value.trim();
  if (!question) {
    if (askLog) askLog.textContent = 'Ask a question first.';
    return;
  }
  let imgs = [];
  try {
    if (chatAttachments.length) {
      const originals = await filesToDataURLs(chatAttachments);
      const thumbs = await makeThumbs(originals, 96, 0.6);
      const viewerImgs = await makeThumbs(originals, 1024, 0.85);
      // Render with light thumbs; store medium previews for viewer in data-full; send originals to API
      imgs = originals;
      const grid = '<div class="user-attach-grid" style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:6px;">'
        + thumbs.map((it, i) => `<img class="user-thumb" src="${it.data}" data-full="${(viewerImgs[i] && viewerImgs[i].data) ? viewerImgs[i].data : it.data}" alt="${escapeHTML(it.name||'image')}" style="width:96px; height:auto; border-radius:8px; border:1px solid rgba(15,23,42,0.12); cursor: zoom-in;" />`).join('')
        + '</div>';
      const body = `<div class="user-text">${escapeHTML(question)}</div>`;
      pendingUserHtml = grid + body;
    }
  } catch {}
  if (Array.isArray(imgs) && imgs.length && pendingUserHtml) {
    appendUserMessage(pendingUserHtml, { html: true });
  } else {
    appendUserMessage(question);
  }
  const payload = {
    question,
    memory_enabled: true,
    formula_mode: document.getElementById('formula-mode').checked,
    strict_docs: document.getElementById('strict-docs')?.checked || false,
    web_enabled: document.getElementById('web-enabled').checked,
    exhaustive: document.getElementById('exhaustive').checked,
    reranker: (() => {
      const on = document.getElementById('reranker-on')?.checked;
      if (!on) return 'off';
      const chosen = (defaultsData && typeof defaultsData.ASK_RERANKER === 'string' && defaultsData.ASK_RERANKER.toLowerCase() !== 'off')
        ? String(defaultsData.ASK_RERANKER)
        : 'minilm';
      return chosen;
    })(),
    top_k: Number(document.getElementById('top-k').value || 10),
    max_batches: Number(document.getElementById('max-batches').value || 6),
    time_budget: Number(document.getElementById('time-budget').value || 120),
    history: chatHistory,
  };
  if (onlyDocSelect && onlyDocSelect.value) {
    payload.only_doc = onlyDocSelect.value;
  }
  if (imgs && imgs.length) payload.images = imgs;
  askQuestion(payload, question);
  const qEl = document.getElementById('question');
  if (qEl) qEl.value = '';
  // Clear attachments after sending
  chatAttachments.length = 0;
  renderChatAttachments();
});

clearChatBtn?.addEventListener('click', () => {
  try { removeTempSpacer(); } catch {}
  chatWindow.innerHTML = '';
  if (askLog) askLog.textContent = 'Conversation cleared.';
  chatHistory.length = 0;
  showEmptyChatTitle();
  if (activeConvId && conversations[activeConvId]) {
    const conv = conversations[activeConvId];
    conv.history = [];
    conv.updatedAt = Date.now();
    conversations[activeConvId] = conv;
    saveConversations();
    renderConversationList();
  }
});

settingsForm?.addEventListener('submit', async (evt) => {
  evt.preventDefault();
  await saveSettingsNow();
});

function collectSettingsPayload() {
  const getEl = (id) => document.getElementById(id);
  return {
    openai_key: getEl('openai-key')?.value || '',
    hf_token: getEl('hf-token')?.value || '',
    serp_key: getEl('serp-key')?.value || '',
    brave_key: getEl('brave-key')?.value || '',
    openai_model: getEl('openai-model')?.value || '',
    hf_model: getEl('hf-model')?.value || '',
    embed_backend: getEl('embed-backend')?.value || 'hf',
    llm_backend: getEl('llm-backend')?.value || 'openai',
    memory_enabled: Boolean(getEl('settings-memory')?.checked),
    memory_tokens: Number(getEl('memory-tokens')?.value || 1200),
    memory_file_mb: Number(getEl('memory-file-mb')?.value || 50),
    openai_tpm: Number(getEl('openai-tpm')?.value || 0),
    openai_rpm: Number(getEl('openai-rpm')?.value || 0),
    ask_char_budget: Number(getEl('ask-char-budget')?.value || 12000),
    ask_max_batches: Number(getEl('ask-max-batches')?.value || 6),
    ask_time_budget: Number(getEl('ask-time-budget')?.value || 120),
    ask_exhaustive: Boolean(getEl('settings-exhaustive')?.checked),
    ask_reranker: (getEl('settings-reranker')?.value || 'minilm'),
    ask_candidates: Number(getEl('ask-candidates')?.value || 300),
    web_provider: (getEl('web-provider')?.value || 'auto'),
  };
}

async function saveSettingsNow() {
  try {
    if (settingsSaveTimer) { clearTimeout(settingsSaveTimer); settingsSaveTimer = null; }
  } catch {}
  const payload = collectSettingsPayload();
  if (settingsStatus) settingsStatus.textContent = 'Saving…';
  try {
    const resp = await fetch(`${API_BASE}/settings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      credentials: 'include',
    });
    if (!resp.ok) {
      const text = await resp.text();
      if (settingsStatus) settingsStatus.textContent = `Error: ${text}`;
      showError(text || 'Failed to save settings');
      return;
    }
    const data = await resp.json();
    if (settingsStatus) settingsStatus.textContent = data.message || 'Saved.';
    if (data.defaults) {
      applyDefaults(data.defaults);
    }
    await refreshSettingsStatus();
  } catch (err) {
    if (settingsStatus) settingsStatus.textContent = 'Error saving settings';
    showError(err);
  }
}

function scheduleSaveSettings(delay = 600) {
  try { if (settingsSaveTimer) clearTimeout(settingsSaveTimer); } catch {}
  settingsSaveTimer = setTimeout(() => { saveSettingsNow(); }, delay);
}

function setupSettingsAutosave() {
  if (!settingsForm) return;
  const inputs = settingsForm.querySelectorAll('input, select');
  inputs.forEach((el) => {
    const tag = el.tagName.toLowerCase();
    const type = (el.getAttribute('type') || '').toLowerCase();
    if (type === 'checkbox' || tag === 'select') {
      el.addEventListener('change', () => { saveSettingsNow(); });
    } else {
      if (type !== 'password') {
        el.addEventListener('input', () => { scheduleSaveSettings(700); });
      }
      el.addEventListener('change', () => { saveSettingsNow(); });
    }
  });
}

function iconCheck() {
  return '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" style="vertical-align:-2px; fill:currentColor"><path d="M9 16.2l-3.5-3.5a1 1 0 1 1 1.4-1.4L9 13.4l8.1-8.1a1 1 0 1 1 1.4 1.4L9 16.2z"/></svg>';
}
function iconBang() {
  return '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" style="vertical-align:-2px; fill:currentColor"><path d="M11 3h2v12h-2V3zm0 14h2v4h-2v-4z"/></svg>';
}
function iconRequired() {
  return '<svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true" style="vertical-align:-2px; stroke:currentColor; fill:none; stroke-width:2"><path d="M12 4v16M4.5 7l15 10M4.5 17l15-10"/></svg>';
}

function renderKeyStatus(keys) {
  if (!keys) return '';
  const parts = [];
  const order = [
    ['openai','OpenAI'],
    ['hf','HF'],
    ['serpapi','SerpAPI'],
    ['brave','Brave'],
  ];
  for (const [k, label] of order) {
    const info = keys[k] || {};
    const ok = info.ok;
    const req = !!info.required;
    let icons = '';
    if (ok === true) icons += iconCheck();
    else if (ok === false) icons += iconBang();
    if (req) icons += ' ' + iconRequired();
    parts.push(`<span class="key-item" data-key="${k}" title="${label}">${label}: ${icons || '—'}</span>`);
  }
  return parts.join('  ');
}

async function refreshSettingsStatus() {
  try {
    const resp = await fetch(`${API_BASE}/settings?verify=1`, { credentials: 'include' });
    if (!resp.ok) { const t = await resp.text(); showError(t || 'Failed to load settings'); return; }
    const data = await resp.json();
    if (data.defaults) {
      applyDefaults(data.defaults);
    }
    const keyStatusEl = document.getElementById('settings-status');
    if (keyStatusEl && data.keys) {
      keyStatusEl.innerHTML = renderKeyStatus(data.keys);
    }
  } catch (err) {
    console.warn('Failed to refresh settings', err);
    showError(err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', refreshSettingsStatus);
} else {
  refreshSettingsStatus();
}

// Initialize settings auto-save listeners once DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupSettingsAutosave);
} else {
  setupSettingsAutosave();
}

// -------------------------------------------------
// Conversations (localStorage based)
function escapeHTML(s) {
  return String(s || '').replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function saveConversations() {
  const payload = { active: activeConvId, conversations };
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    return;
  } catch (e1) {
    // Quota exceeded or serialization too large — compact and retry
    try {
      const compact = (src) => {
        const out = {};
        for (const [id, conv] of Object.entries(src || {})) {
          const hist = Array.isArray(conv.history) ? conv.history.slice(-60) : [];
          const trimmed = hist.map((it) => {
            const qhtml = (it.q_html && typeof it.q_html === 'string' && it.q_html.length <= 6000) ? it.q_html : null;
            let ah = it.a_html || '';
            let am = it.a_markdown || it.a || '';
            if (typeof ah === 'string' && ah.length > 20000) ah = ah.slice(0, 20000);
            if (typeof am === 'string' && am.length > 16000) am = am.slice(0, 16000);
            return { q: it.q || '', q_html: qhtml, a: am || '', a_markdown: am || '', a_html: ah || '' };
          });
          out[id] = { id: conv.id, title: conv.title, createdAt: conv.createdAt, updatedAt: conv.updatedAt, history: trimmed };
        }
        return out;
      };
      const small = { active: activeConvId, conversations: compact(conversations) };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(small));
      return;
    } catch (e2) {
      try {
        // Last resort: drop previews and keep only last 20 turns
        const minimal = {};
        for (const [id, conv] of Object.entries(conversations || {})) {
          const hist = Array.isArray(conv.history) ? conv.history.slice(-20) : [];
          const minHist = hist.map((it) => ({ q: it.q || '', a: (it.a_markdown || it.a || '').slice(0, 8000), a_markdown: (it.a_markdown || it.a || '').slice(0, 8000), a_html: (it.a_html || '').slice(0, 12000) }));
          minimal[id] = { id: conv.id, title: conv.title, createdAt: conv.createdAt, updatedAt: conv.updatedAt, history: minHist };
        }
        localStorage.setItem(STORAGE_KEY, JSON.stringify({ active: activeConvId, conversations: minimal }));
      } catch (e3) {
        try { showError('Storage is full — could not save conversations.'); } catch {}
      }
    }
  }
}

function renderConversationList() {
  if (!convListEl) return;
  const items = Object.values(conversations).sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
  convListEl.innerHTML = items
    .map(
      (c) =>
        `<li data-id="${c.id}" class="${c.id === activeConvId ? 'active' : ''}"><span class="conv-title">${escapeHTML(
          c.title || 'New chat'
        )}</span><span class="conv-actions"><button type="button" class="conv-rename" title="Rename" aria-label="Rename chat" onclick="return window._uiRenameConv?.('${c.id}')"><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M4 16.25V20h3.75L18.81 8.94a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0L4 16.25z" fill="currentColor"/></svg></button><button type="button" class="conv-delete" title="Delete" aria-label="Delete chat" onclick="return window._uiDeleteConv?.('${c.id}')"><svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M6 7h12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M10 11v6M14 11v6" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path d="M9 7l1-2h4l1 2M8 7l1 12a2 2 0 0 0 2 2h2a2 2 0 0 0 2-2l1-12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></button></span></li>`
    )
    .join('');
  // Attach explicit handlers to ensure clicks are caught even if delegation fails
  convListEl.querySelectorAll('li').forEach((li) => {
    const id = li.getAttribute('data-id');
    const del = li.querySelector('button.conv-delete');
    const ren = li.querySelector('button.conv-rename');
    if (del && !del.dataset.bound) {
      del.dataset.bound = '1';
      del.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        if (!id) return;
        const sure = confirm('Delete this chat?');
        if (sure) deleteConversation(id);
      });
    }
    if (ren && !ren.dataset.bound) {
      ren.dataset.bound = '1';
      ren.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        if (!id) return;
        const current = conversations[id]?.title || '';
        const next = prompt('Rename chat', current);
        if (next !== null) renameConversation(id, String(next).trim());
      });
    }
  });
}

function renderMarkdownToHtml(markdown) {
  if (!markdown) return '';
  if (window.marked && typeof window.marked.parse === 'function') {
    const mathSegments = [];
    const stash = (match) => {
      const key = `@@MATH${mathSegments.length}@@`;
      mathSegments.push(match);
      return key;
    };
    let source = String(markdown);
    const mathPatterns = [
      /(\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\})/g,
      /\$\$([\s\S]+?)\$\$/g,
      /\\\[[\s\S]+?\\\]/g,
      /\\\([\s\S]+?\\\)/g,
      /\$[^$]+\$/g,
    ];
    for (const pattern of mathPatterns) {
      source = source.replace(pattern, stash);
    }
    let html = window.marked.parse(source);
    html = html.replace(/@@MATH(\d+)@@/g, (_, idx) => mathSegments[idx] || '');
    return html;
  }
  return escapeHTML(markdown).replace(/\n/g, '<br />');
}

function stripHTML(html) {
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  return tmp.textContent || tmp.innerText || '';
}

async function maybeUpdateTitleFrom(text) {
  try {
    const id = activeConvId;
    if (!id || !conversations[id]) return;
    const current = conversations[id].title || '';
    // Only auto-title if the title is default-ish
    const isDefault = !current || current === 'New chat' || /^Summary:/i.test(current) || current.length < 4;
    if (!isDefault) return;
    // Build a richer summary from conversation so far
    let combined = '';
    try {
      const hist = conversations[id].history || chatHistory || [];
      const recent = hist.slice(-6);
      const parts = [];
      for (const turn of recent) {
        const q = (turn.q || '').trim();
        const a = (turn.a_markdown || turn.a || '').trim();
        if (q) parts.push('Q: ' + q);
        if (a) parts.push('A: ' + a);
      }
      combined = parts.join('\n');
    } catch {}
    const resp = await fetch(`${API_BASE}/chat/title`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: String((combined || text || '')).slice(0, 4000) }),
      credentials: 'include',
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const title = (data.title || '').trim();
    if (title) renameConversation(id, title);
  } catch {}
}

function recordConversationEntry(question, answerHtml, answerMarkdown, questionHtml) {
  // Allow saving turns with no answer (e.g., user cancelled) so they persist across refresh
  if (!question && !answerHtml && !answerMarkdown) return;
  const htmlValue = (answerHtml && answerHtml.trim())
    ? answerHtml
    : (answerMarkdown ? renderMarkdownToHtml(answerMarkdown) : '');
  const markdownValue = (answerMarkdown && answerMarkdown.trim())
    ? answerMarkdown
    : (htmlValue ? stripHTML(htmlValue) : '');
  const entry = {
    q: question,
    q_html: questionHtml || null,
    a: markdownValue,
    a_markdown: answerMarkdown || markdownValue,
    a_html: htmlValue,
  };
  chatHistory.push(entry);
  if (activeConvId && conversations[activeConvId]) {
    const conv = conversations[activeConvId];
    conv.history = [...chatHistory];
    if (!conv.title || conv.title === 'New chat') conv.title = question.slice(0, 72);
    conv.updatedAt = Date.now();
    conversations[activeConvId] = conv;
    saveConversations();
    renderConversationList();
  }
}

function loadConversation(id) {
  if (chatBody) { chatBody.style.visibility = 'hidden'; }
  chatWindow.innerHTML = '';
  chatHistory.length = 0;
  if (!id || !conversations[id]) return;
  activeConvId = id;
  const hist = conversations[id].history || [];
  const normalized = [];
  let lastEl = null;
  for (const item of hist) {
    const question = item.q || '';
    const qhtml = item.q_html || '';
    const markdown = item.a_markdown || item.a || '';
    const html = item.a_html || (markdown ? renderMarkdownToHtml(markdown) : '');
    if (qhtml) {
      lastEl = appendUserMessage(qhtml, { scroll: false, animate: false, html: true });
    } else if (question) {
      lastEl = appendUserMessage(question, { scroll: false, animate: false });
    }
    if (html) lastEl = appendBotMessage(html, { scroll: false, typeset: false, animate: false });
    normalized.push({ q: question, q_html: qhtml || null, a: markdown, a_markdown: markdown, a_html: html });
  }
  chatHistory.push(...normalized);
  conversations[id].history = normalized;
  renderConversationList();
  ensureBottomSentinel();
  if (lastEl) {
    scrollMessageStart(lastEl);
    requestAnimationFrame(() => scrollMessageStart(lastEl));
    typesetMath().then(() => {
      scrollMessageStart(lastEl);
      requestAnimationFrame(() => scrollMessageStart(lastEl));
    });
    setTimeout(() => scrollMessageStart(lastEl), 80);
    setTimeout(() => scrollMessageStart(lastEl), 180);
  } else {
    scrollChatToBottom();
  }
  // Re-bind sentinel after replacing chat content
  initScrollObserver();
  saveConversations();
  if (normalized.length === 0) showEmptyChatTitle();
  if (chatBody) { setTimeout(() => { chatBody.style.visibility = ''; updateScrollDownBtn(); }, 0); }
}

function generateId() {
  return 'c_' + Math.random().toString(36).slice(2, 10);
}

function createConversation(title = 'New chat') {
  const id = generateId();
  conversations[id] = { id, title, history: [], createdAt: Date.now(), updatedAt: Date.now() };
  activeConvId = id;
  saveConversations();
  renderConversationList();
  loadConversation(id);
  setRandomPlaceholder(true);
}

function deleteConversation(id) {
  if (!conversations[id]) return;
  delete conversations[id];
  if (activeConvId === id) {
    const ids = Object.keys(conversations);
    activeConvId = ids.length ? ids[0] : null;
  }
  saveConversations();
  renderConversationList();
  if (!activeConvId) {
    // If no conversations remain, create one and switch to chat
    createConversation();
  } else {
    loadConversation(activeConvId);
  }
}

function renameConversation(id, title) {
  if (!conversations[id]) return;
  conversations[id].title = title || 'Untitled';
  conversations[id].updatedAt = Date.now();
  saveConversations();
  renderConversationList();
}

// Global helpers for inline handlers (defensive against event delegation issues)
window._uiDeleteConv = (id) => {
  try {
    if (!id || !conversations[id]) return false;
    const sure = confirm('Delete this chat?');
    if (sure) deleteConversation(id);
  } catch {}
  return false;
};

window._uiRenameConv = (id) => {
  try {
    if (!id || !conversations[id]) return false;
    const current = conversations[id]?.title || '';
    const next = prompt('Rename chat', current);
    if (next !== null) renameConversation(id, String(next).trim());
  } catch {}
  return false;
};

function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      conversations = data.conversations || {};
      activeConvId = data.active || null;
    }
  } catch {}
  if (!activeConvId || !conversations[activeConvId]) {
    const ids = Object.keys(conversations);
    activeConvId = ids.length ? ids[0] : null;
  }
  if (!activeConvId) createConversation();
  renderConversationList();
  loadConversation(activeConvId);
}

newChatBtn?.addEventListener('click', () => createConversation());

convListEl?.addEventListener('click', (evt) => {
  // Normalize target to an Element (buttons contain emoji text nodes)
  let target = evt.target;
  if (!(target instanceof Element)) {
    target = target && target.parentElement ? target.parentElement : null;
  }
  if (!target) return;
  const li = target.closest('li');
  if (!li) return;
  const id = li.getAttribute('data-id');
  if (!id) return;
  if (target.closest('button.conv-delete')) {
    evt.stopPropagation(); evt.preventDefault();
    const sure = confirm('Delete this chat?');
    if (sure) deleteConversation(id);
    return;
  }
  if (target.closest('button.conv-rename')) {
    evt.stopPropagation(); evt.preventDefault();
    const current = conversations[id]?.title || '';
    const next = prompt('Rename chat', current);
    if (next !== null) renameConversation(id, String(next).trim());
    return;
  }
  loadConversation(id);
  if (sidebarMedia.matches) setSidebarOpen(false);
});

// Extra safeguard: capture pointer events early to ensure buttons work
convListEl?.addEventListener('pointerdown', (evt) => {
  let t = evt.target;
  if (!(t instanceof Element)) {
    t = t && t.parentElement ? t.parentElement : null;
  }
  if (!t) return;
  if (t.closest('button.conv-delete') || t.closest('button.conv-rename')) {
    // Prevent row selection on button press
    evt.stopPropagation();
  }
}, true);

// Initialize conversations after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    loadConversations();
    handleSidebarViewportChange();
    updateControlsDrawer(true);
    refreshLibrary();
    setRandomPlaceholder(true);
    measureLayoutVars();
    positionScrollButton();
    initScrollObserver();
    initDevTools();
    initImageViewerZoom();
    try { updateScrollDownBtn(); } catch {}
    try {
      const ftr = document.querySelector('.chat-footer');
      if (window.ResizeObserver && ftr) {
        const roBtn = new ResizeObserver(() => positionScrollButton());
        roBtn.observe(ftr);
      }
    } catch {}
  });
} else {
  loadConversations();
  handleSidebarViewportChange();
  updateControlsDrawer(true);
  refreshLibrary();
  setRandomPlaceholder(true);
  measureLayoutVars();
  positionScrollButton();
  initScrollObserver();
  initDevTools();
  initImageViewerZoom();
  try { updateScrollDownBtn(); } catch {}
}

// Zoom & pan interactions for page/image viewer
let __viewer = { s: 1 };
function applyViewerTransform() {
  try {
    if (!pageViewerImg) return;
    const s = Math.max(0.25, Math.min(12, (__viewer.s || 1)));
    pageViewerImg.style.maxWidth = '90vw';
    pageViewerImg.style.maxHeight = '90vh';
    pageViewerImg.style.transformOrigin = 'center center';
    pageViewerImg.style.transform = `scale(${s})`;
  } catch {}
}
function resetViewerTransform() { __viewer = { s: 1 }; applyViewerTransform(); }
function initImageViewerZoom() {
  if (!pageViewerImg) return;
  try { pageViewerImg.style.transition = 'transform .05s linear'; } catch {}
  pageViewerImg.addEventListener('wheel', (e) => {
    try {
      e.preventDefault();
      // Scroll up = zoom in, scroll down = zoom out
      const dir = e.deltaY < 0 ? 1.12 : 0.89;
      __viewer.s = Math.max(0.25, Math.min(12, (__viewer.s || 1) * dir));
      applyViewerTransform();
    } catch {}
  }, { passive: false });
  pageViewerImg.addEventListener('dblclick', () => { resetViewerTransform(); });
}

function initDevTools() {
  const statusEl = document.getElementById('dev-status');
  const consoleInput = document.getElementById('dev-console-input');
  const consoleRun = document.getElementById('dev-console-run');
  const consoleOut = document.getElementById('dev-console-output');
  async function callAdmin(path) {
    const url = `${API_BASE}/admin/${path}`;
    try {
      const resp = await fetch(url, { method: 'POST', credentials: 'include' });
      const text = await resp.text();
      if (statusEl) statusEl.textContent = (resp.ok ? 'OK: ' : 'Error: ') + text;
    } catch (err) {
      if (statusEl) statusEl.textContent = `Error: ${err}`;
    }
  }
  async function refreshAdminStatus() {
    try {
      const resp = await fetch(`${API_BASE}/admin/status`, { credentials: 'include' });
      const text = await resp.text();
      let obj = text; try { obj = JSON.parse(text); } catch {}
      if (statusEl) {
        if (typeof obj === 'object') {
          const ns = Array.isArray(obj.namespaces) ? obj.namespaces.length : 0;
          statusEl.textContent = `Docs: ${obj.total_rows || 0} rows across ${ns} namespaces • Users: ${obj.users || 0} • Memory files: ${obj.memory_files || 0}`;
        } else {
          statusEl.textContent = String(obj);
        }
      }
    } catch (err) {
      if (statusEl) statusEl.textContent = `Error loading status: ${err}`;
    }
  }
  async function runConsoleCommand() {
    if (!consoleInput || !consoleOut) return;
    const raw = String(consoleInput.value || '').trim();
    if (!raw) return;
    const show = (obj, ok=true) => {
      try {
        const text = (typeof obj === 'string') ? obj : JSON.stringify(obj, null, 2);
        consoleOut.textContent = text;
        if (statusEl) statusEl.textContent = ok ? 'OK' : 'Error';
      } catch { consoleOut.textContent = String(obj); }
    };
    try {
      const parts = raw.split(/\s+/);
      const first = (parts[0] || '').toLowerCase();
      const second = (parts[1] || '').replace(/^\//, '').toLowerCase();
      const third = (parts[2] || '').toLowerCase();
      const isSlashForm = first.startsWith('/');
      const op = isSlashForm ? first.slice(1) : first; // accept with or without leading '/'

      const doGet = async (p) => {
        const url = `${API_BASE}/admin/${p}`;
        const resp = await fetch(url, { credentials: 'include' });
        const text = await resp.text();
        let obj = text; try { obj = JSON.parse(text); } catch {}
        show(obj, resp.ok);
      };
      const doPost = async (p) => {
        const url = `${API_BASE}/admin/${p}`;
        const resp = await fetch(url, { method: 'POST', credentials: 'include' });
        const text = await resp.text();
        let obj = text; try { obj = JSON.parse(text); } catch {}
        show(obj, resp.ok);
      };

      // Friendly shorthands
      if (!op) { show('Enter a command. Example: get usage'); return; }
      if (op === 'usage') return void doGet('usage');
      if (op === 'users') return void doGet('users');
      if (op === 'rebuild' || op === 'rebuild_index') return void doPost('rebuild_index');
      if (op === 'scan') {
        const name = parts.slice(1).join(' ').trim();
        if (!name) { show('Usage: scan <document name>'); return; }
        const resp = await fetch(`${API_BASE}/admin/scan_doc?name=${encodeURIComponent(name)}`, { credentials: 'include' });
        const text = await resp.text(); let obj = text; try { obj = JSON.parse(text); } catch {}
        show(obj, resp.ok); return;
      }
      if (op === 'purge') {
        const name = parts.slice(1).join(' ').trim();
        if (!name) { show('Usage: purge <document name>'); return; }
        const resp = await fetch(`${API_BASE}/admin/purge_doc?name=${encodeURIComponent(name)}`, { method: 'POST', credentials: 'include' });
        const text = await resp.text(); let obj = text; try { obj = JSON.parse(text); } catch {}
        show(obj, resp.ok); return;
      }
      if (op === 'clear') {
        let endpoint = null;
        if (second === 'settings') endpoint = 'clear_settings';
        else if (second === 'memory') endpoint = 'clear_memory';
        else if (second === 'all') endpoint = 'clear_all';
        if (!endpoint) { show('Usage: clear settings|memory|all'); return; }
        return void doPost(endpoint);
      }

      // Verb + arg forms
      if (op === 'get') {
        if (!second) { show('Usage: get <path> (e.g., get usage)'); return; }
        return void doGet(second);
      }
      if (op === 'post') {
        if (!second) { show('Usage: post <path> (e.g., post rebuild_index)'); return; }
        return void doPost(second);
      }

      show('Unknown command. Try: get usage, get users, rebuild, clear settings');
    } catch (err) {
      show(String(err), false);
    }
  }
  document.getElementById('dev-clear-settings')?.addEventListener('click', () => callAdmin('clear_settings'));
  document.getElementById('dev-clear-memory')?.addEventListener('click', () => callAdmin('clear_memory'));
  document.getElementById('dev-clear-all')?.addEventListener('click', () => callAdmin('clear_all'));
  document.getElementById('dev-purge-libraries')?.addEventListener('click', async () => {
    const sure = confirm('Purge ALL users\' libraries across all namespaces? This cannot be undone.');
    if (!sure) return;
    await callAdmin('purge_all_libraries');
    try { await refreshLibrary(); } catch {}
  });
  // (Removed) Attach sources helper; reingest instead
  // Update status initially and on a timer
  refreshAdminStatus();
  try { setInterval(refreshAdminStatus, 8000); } catch {}
  consoleRun?.addEventListener('click', runConsoleCommand);
  consoleInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); runConsoleCommand(); }
  });
  document.getElementById('dev-clear-chats')?.addEventListener('click', () => {
    try {
      const keys = Object.keys(localStorage);
      let n = 0;
      for (const k of keys) {
        if (k.startsWith('anag_conversations_')) { localStorage.removeItem(k); n++; }
      }
      if (statusEl) statusEl.textContent = `Cleared ${n} conversation storages locally.`;
      loadConversations();
    } catch (err) {
      if (statusEl) statusEl.textContent = `Error clearing chats: ${err}`;
    }
  });
}

// Global defensive delegation: ensure rename/delete always work
document.addEventListener('click', (evt) => {
  let t = evt.target;
  if (!(t instanceof Element)) t = t?.parentElement || null;
  if (!t) return;
  const btn = t.closest('button.conv-delete, button.conv-rename');
  if (!btn) return;
  const li = btn.closest('li[data-id]');
  const id = li?.getAttribute('data-id');
  if (!id) return;
  evt.preventDefault();
  evt.stopPropagation();
  if (btn.classList.contains('conv-delete')) {
    const sure = confirm('Delete this chat?');
    if (sure) deleteConversation(id);
  } else if (btn.classList.contains('conv-rename')) {
    const current = conversations[id]?.title || '';
    const next = prompt('Rename chat', current);
    if (next !== null) renameConversation(id, String(next).trim());
  }
}, true);

window.addEventListener('resize', () => {
  handleSidebarViewportChange();
  updateControlsDrawer();
  measureLayoutVars();
  positionScrollButton();
  initScrollObserver();
  try { updateScrollDownBtn(); } catch {}
  // Keep temporary padding proportional and alignment stable on resize
  if (pinnedTopMessage) {
    try { alignMessageTopWithSpacer(pinnedTopMessage); } catch {}
  } else if (tempBottomSpacer && tempBottomSpacer.isConnected) {
    // If we don't have a pinned element but spacer exists, cap it to the new proportion
    try {
      tempBottomSpacer.style.height = Math.max(0, Math.min(parseInt(tempBottomSpacer.style.height || '0', 10) || 0, getTempSpacerCap())) + 'px';
    } catch {}
  }
});
// No additional rotation on focus/click — only on initialization/new chat.
// Scroll button visibility and behavior
chatBody?.addEventListener('scroll', () => { try { updateScrollDownBtn(); } catch {} });
window.addEventListener('scroll', () => { try { updateScrollDownBtn(); } catch {} }, { passive: true });
scrollDownBtn?.addEventListener('click', (e) => {
  e.preventDefault();
  try {
    const last = chatWindow?.querySelector('.message:last-of-type');
    if (last) {
      scrollMessageStart(last);
      requestAnimationFrame(() => scrollMessageStart(last));
    } else {
      scrollChatToBottom();
    }
  } catch { scrollChatToBottom(); }
});
// Make sure button is correctly placed after fonts/layout settle
setTimeout(positionScrollButton, 300);
window.addEventListener('orientationchange', () => updateControlsDrawer(true));

refreshLibraryBtn?.addEventListener('click', () => refreshLibrary());

async function applyLibraryDeletes(names) {
  try {
    librarySaveController = new AbortController();
    const resp = await fetch(`${API_BASE}/library/delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
      credentials: 'include',
      signal: librarySaveController.signal,
    });
    if (!resp.ok) {
      const t = await resp.text();
      alert('Failed to save changes: ' + t);
      return false;
    }
    return true;
  } catch (err) {
    if (err && err.name === 'AbortError') return 'aborted';
    alert('Failed to save changes: ' + err);
    return false;
  }
}

saveLibraryBtn?.addEventListener('click', async () => {
  if (pendingRemovals.size === 0) return;
  const names = Array.from(pendingRemovals);
  const ok = confirm(`Remove ${names.length} selected book(s) and rebuild the index?`);
  if (!ok) return;
  // Normalize to base filenames to match server keys
  const normalized = names.map((n) => {
    try { return n.split(/[/\\\\]/).pop(); } catch { return n; }
  });
  saveLibraryBtn.disabled = true;
  saveLibraryBtn.textContent = 'Saving…';
  if (cancelLibraryBtn) cancelLibraryBtn.style.display = '';
  const done = await applyLibraryDeletes(normalized);
  if (done) {
    pendingRemovals.clear();
    updateLibraryActionButtons();
    await refreshLibrary();
  }
  saveLibraryBtn.disabled = false;
  if (cancelLibraryBtn) cancelLibraryBtn.style.display = 'none';
  if (done === 'aborted') {
    saveLibraryBtn.textContent = 'Save';
    return;
  }
});

cancelLibraryBtn?.addEventListener('click', () => {
  if (!librarySaveController) return;
  try { librarySaveController.abort(); } catch {}
});

clearLibraryBtn?.addEventListener('click', async () => {
  const ok1 = confirm('Clear ALL books from the library and rebuild the index?');
  if (!ok1) return;
  const ok2 = confirm('Are you absolutely sure? This cannot be undone.');
  if (!ok2) return;
  try {
    const resp = await fetch(`${API_BASE}/library/clear`, { method: 'POST', credentials: 'include' });
    if (!resp.ok) { alert('Failed to clear library: ' + (await resp.text())); return; }
    pendingRemovals.clear();
    updateLibraryActionButtons();
    await refreshLibrary();
  } catch (err) {
    alert('Failed to clear library: ' + err);
  }
});

libraryGrid?.addEventListener('click', (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.classList.contains('library-remove')) {
    const name = target.getAttribute('data-doc');
    if (!name) return;
    if (pendingRemovals.has(name)) {
      pendingRemovals.delete(name);
    } else {
      pendingRemovals.add(name);
    }
    const card = target.closest('.book-card');
    if (card) card.classList.toggle('pending-remove');
    target.textContent = pendingRemovals.has(name) ? 'Undo' : 'Remove';
    updateLibraryActionButtons();
  }
});

// Clear selection button
const clearSelectionBtn = document.getElementById('clear-selection');
clearSelectionBtn?.addEventListener('click', () => {
  selectedFiles = [];
  setInputFiles([]);
  const input = document.getElementById('upload-input');
  if (input) input.value = '';
  renderFileList([], { hideProgress: true });
});

// Fallback: delegate clear button clicks (in case of dynamic reflow)
document.addEventListener('click', (evt) => {
  const t = evt.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.id === 'clear-selection') {
    evt.preventDefault();
    selectedFiles = [];
    setInputFiles([]);
    const input = document.getElementById('upload-input');
    if (input) input.value = '';
    renderFileList([], { hideProgress: true });
  }
});

// Enter to send (Shift+Enter for newline)
document.addEventListener('keydown', (evt) => {
  const target = evt.target;
  if (!(target instanceof HTMLTextAreaElement)) return;
  if (target.id !== 'question') return;
  if (evt.key === 'Enter' && !evt.shiftKey) {
    evt.preventDefault();
    askForm?.dispatchEvent(new Event('submit', { cancelable: true }));
  }
});

document.addEventListener('keydown', (evt) => {
  if (evt.key === 'Escape' && document.body.classList.contains('sidebar-open')) {
    setSidebarOpen(false);
  }
});
