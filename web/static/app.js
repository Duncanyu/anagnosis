const tabs = document.querySelectorAll('.anag-tabs button');
try { console.info('[anag-ui] app.js v45 loaded (workspace)'); } catch {}
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

  setValue('embed-backend', defaults.EMBED_BACKEND);
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

// Actions popover: show panel on button click, not a toggle
const actionsPopover = document.getElementById('actions-popover');
const actionsTrigger = document.getElementById('actions-trigger');
function positionActionsPopover() {
  try {
    if (!actionsPopover || !actionsTrigger) return;
    const r = actionsTrigger.getBoundingClientRect();
    const panel = actionsPopover.querySelector('.ex-panel');
    const pad = 8;
    const x = Math.max(8, Math.min((r.left + (r.width/2) - 140), window.innerWidth - 300));
    const y = Math.max(8, r.top - (panel?.offsetHeight || 400) - 12);
    actionsPopover.style.left = x + 'px';
    actionsPopover.style.top = y + 'px';
  } catch {}
}
function showActionsPopover(show) {
  try {
    if (!actionsPopover) return;
    actionsPopover.style.display = show ? 'block' : 'none';
    if (show) {
      positionActionsPopover();
      window.addEventListener('resize', positionActionsPopover);
      window.addEventListener('scroll', positionActionsPopover, true);
      document.addEventListener('click', handleActionsOutside, true);
    } else {
      window.removeEventListener('resize', positionActionsPopover);
      window.removeEventListener('scroll', positionActionsPopover, true);
      document.removeEventListener('click', handleActionsOutside, true);
    }
  } catch {}
}
function handleActionsOutside(e){
  try {
    if (!actionsPopover || actionsPopover.style.display === 'none') return;
    const t = e.target;
    if (!(t instanceof Node)) return;
    if (!actionsPopover.contains(t) && !actionsTrigger.contains(t)) {
      showActionsPopover(false);
    }
  } catch {}
}
actionsTrigger?.addEventListener('click', (e) => {
  e.preventDefault();
  e.stopPropagation();
  try {
    const isVisible = actionsPopover && actionsPopover.style.display === 'block';
    showActionsPopover(!isVisible);
  } catch {}
});

// Handle action button clicks
actionsPopover?.addEventListener('click', async (e) => {
  try {
    const btn = e.target.closest('.action-btn');
    if (!btn) return;
    e.preventDefault();
    e.stopPropagation();
    
    const actionId = btn.getAttribute('data-action');
    const paramsStr = btn.getAttribute('data-params');
    let params = null;
    if (paramsStr) {
      try { params = JSON.parse(paramsStr); } catch {}
    }
    
    if (actionId) {
      showActionsPopover(false);
      
      // Actions that don't need immediate user message
      const noImmediateMessageActions = new Set([
        'enable_web',
        'broaden_docs',
        'select_doc',
        'set_top_k',
        'toggle_reranker',
        'strict_docs_only',
        'exhaustive_search',
        'upload_docs'
      ]);
      
      // Show user message for most actions (so there's context for the action)
      if (!noImmediateMessageActions.has(actionId)) {
        try {
          const actionLabel = btn.querySelector('.action-label')?.textContent?.trim() || actionId;
          if (actionLabel) {
            appendUserMessage(actionLabel);
            // Persist immediately so it's not lost on refresh; mark as pending to complete later
            try { 
              recordConversationEntry(actionLabel, '', '', null); 
              window.__pendingActionQuestion = actionLabel; 
            } catch {}
          }
        } catch {}
      }
      
      // Special handling for formula sheet - enable formula mode first
      if (actionId === 'generate_formula_sheet') {
        const formulaCheckbox = document.getElementById('formula-mode');
        if (formulaCheckbox && !formulaCheckbox.checked) {
          formulaCheckbox.checked = true;
        }
      }
      
      await performAction(actionId, params);
    }
  } catch (err) {
    console.error('Action click error:', err);
    showError(err && err.message ? err.message : String(err));
  }
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
    // Helper to complete last pending turn or create a new one
    const updateOrRecord = (q, ansHtml, ansMd) => {
      try {
        const last = chatHistory[chatHistory.length - 1];
        if (last && String(last.q || '') === String(q || '') && !(last.a_markdown || last.a || last.a_html)) {
          last.a_html = ansHtml || '';
          last.a_markdown = ansMd || (ansHtml ? stripHTML(ansHtml) : '');
          last.a = last.a_markdown;
          if (activeConvId && conversations[activeConvId]) {
            const conv = conversations[activeConvId];
            conv.history = [...chatHistory];
            conv.updatedAt = Date.now();
            conversations[activeConvId] = conv;
            saveConversations();
            renderConversationList();
          }
        } else {
          recordConversationEntry(q, ansHtml || '', ansMd || '');
        }
      } catch { recordConversationEntry(q, ansHtml || '', ansMd || ''); }
    };
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

  let lastUpdate = Date.now();
  let staleWarningShown = false;

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
      
      // Check for stale updates (backend might be stuck on a heavy operation)
      const serverLastUpdate = Number(data.last_update || 0) * 1000; // Convert to ms
      const now = Date.now();
      const timeSinceUpdate = now - Math.max(lastUpdate, serverLastUpdate);
      
      // If no update for 30+ seconds, show a processing indicator
      if (timeSinceUpdate > 30000 && !staleWarningShown && data.status === 'running') {
        const logs = Array.isArray(data.logs) ? data.logs : [];
        const lastLog = logs[logs.length - 1] || '';
        if (lastLog && !lastLog.includes('(processing large page)')) {
          ingestLog.textContent += '\n⏳ Processing large page, this may take a few minutes...';
          scrollIngestToBottom();
          staleWarningShown = true;
        }
      }
      
      // Reset stale warning if we get a new update
      if (serverLastUpdate > lastUpdate) {
        lastUpdate = serverLastUpdate;
        staleWarningShown = false;
      }
      
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
          const processing = timeSinceUpdate > 15000 ? ' 🔄' : '';
          statusEl.textContent = `Parsing… ${Math.round(pctDisplay)}%  (${pagesDone} / ${pagesTotal} pages)${processing}`;
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

function maybeAppendTrace(trace, targetEl = null) {
  try {
    const html = renderTraceHTML(trace);
    if (!html) {
      // If no trace payload but a target element exists, move any existing trace under it
      try {
        if (targetEl && chatWindow) {
          const existing = chatWindow.querySelector('.trace-inline, details.trace');
          if (existing) {
            // Ensure only one trace: remove any other extra traces first
            chatWindow.querySelectorAll('.trace-inline, details.trace').forEach((el, idx) => {
              if (idx > 0) el.remove();
            });
            targetEl.appendChild(existing);
          }
        }
      } catch {}
      return;
    }
    // Ensure only the latest message has a trace chip
    try {
      chatWindow?.querySelectorAll('.trace-inline, details.trace').forEach((el) => el.remove());
    } catch {}
    const lastBot = targetEl || chatWindow?.querySelector('.message.bot:last-of-type');
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

// Render assistant-suggested actions (buttons) and optional next_steps
function maybeRenderActions(actions, next_steps, targetBotMessage = null) {
  try {
    const hasActions = Array.isArray(actions) && actions.length > 0;
    const hasNextSteps = Array.isArray(next_steps) && next_steps.length > 0;
    
    if (!hasActions && !hasNextSteps) {
      return;
    }
    
    console.log('[UI] Rendering actions:', hasActions ? actions.length : 0, 'next steps:', hasNextSteps ? next_steps.length : 0);
    
    // Remove any existing action blocks (only show on latest message)
    try { chatWindow?.querySelectorAll('.action-inline').forEach((el) => el.remove()); } catch (e) {}
    
    // Use provided element or find the last bot message
    const lastBot = targetBotMessage || chatWindow?.querySelector('.message.bot:last-of-type');
    
    if (!lastBot) {
      console.warn('[UI] No bot message found to attach actions to');
      return;
    }
    
    const esc = escapeHTML;
    const wrap = document.createElement('div');
    wrap.className = 'action-inline';
    const parts = [];
    const btns = [];
  // Control-style actions that should not always appear in the main row
  const controlActionIds = new Set(['enable_web']);
    
    // Process actions only if they exist
    if (hasActions) {
      for (const rawA of actions) {
      try {
        if (!rawA) {
          continue;
        }
        // Normalize string items to objects: 'enable_web' -> { id: 'enable_web', label: 'Enable web' }
        let a = rawA;
        if (typeof rawA === 'string') {
          a = { id: rawA, label: rawA };
        }
        if (typeof a !== 'object' || Array.isArray(a)) {
          continue;
        }
        if (a.available === false) {
          continue;
        }
        const id = String(a.id || '').trim();
        if (!id) {
          continue;
        }
  const label = String(a.label || id).trim();
        const desc = String(a.description || '');
        const recommended = !!a.recommended;
        const cls = recommended ? 'assistant-action primary' : 'assistant-action secondary';
        // If params exist, serialize to a JSON string and attach as data attribute
        let paramAttr = '';
        try {
          if (a.params && typeof a.params === 'object') {
            paramAttr = ` data-action-params="${esc(encodeURIComponent(JSON.stringify(a.params)))}"`;
          }
        } catch (e) {
          paramAttr = '';
        }
        // Skip control actions in the always-visible main button row (they can still appear in next steps)
        if (controlActionIds.has(id)) {
          continue;
        }
  btns.push(`<button type="button" class="${cls}" data-action-id="${esc(id)}" title="${esc(desc)}"${paramAttr}>${esc(label)}</button>`);
      } catch (e) {
        console.error('[UI] Error rendering action:', e);
        try { console.warn('render action failed', e); } catch {}
      }
    }
    }
    
    // Temporary: only show suggested actions in the Next Steps section
    // if (btns.length) {
    //   parts.push(`<div class="action-buttons" style="display:flex;gap:8px;flex-wrap:wrap">${btns.join('')}</div>`);
    // }
    // Next steps block - always show as actionable buttons
    if (Array.isArray(next_steps) && next_steps.length) {
      // De-duplicate and add a touch of variety by avoiding the most recent identical action
      try {
        next_steps = Array.from(new Set(next_steps || []));
        if (window.__recentActionId) {
          next_steps = next_steps.filter((id) => id !== window.__recentActionId);
        }
        // If variety is low, inject alternates not already present (aim for 5-6 actions)
        const pool = ['broaden_docs', 'set_top_k', 'select_doc', 'toggle_reranker', 'exhaustive_search', 'simplify_explanation', 'cite_sources_only', 'followup_questions', 'mindmap_outline', 'compare_sources'];
        for (const cand of pool) {
          if (next_steps.length >= 6) break;
          if (!next_steps.includes(cand)) next_steps.push(cand);
        }
      } catch {}
      const stepBtns = next_steps.map((s, idx) => {
        const actionId = String(s || '').trim();
        
        // Map action IDs to user-friendly labels
        const actionLabels = {
          'enable_web': 'Regenerate with Web Search',
          'upload_docs': 'Upload Documents', 
          'expand_detail': 'Expand Answer',
          'generate_formula_sheet': 'Generate Formula Sheet',
          'broaden_docs': 'Broaden Search',
          'select_doc': 'Focus on Document',
          'set_top_k': 'Adjust Search Depth',
          'toggle_reranker': 'Regenerate with Reranker',
          'strict_docs_only': 'Regenerate RAG-Only',
          'exhaustive_search': 'Regenerate Exhaustive Search',
          'summarize_docs': 'Summarize Documents',
          'generate_quiz': 'Generate Quiz',
          'compare_sources': 'Compare Sources',
          'simplify_explanation': 'Simplify Explanation',
          'cite_sources_only': 'List Citations',
          'mindmap_outline': 'Create Mind Map',
          'followup_questions': 'Suggest Follow-ups',
          'translate_answer': 'Translate Answer',
          'debug_reasoning_trace': 'Show Reasoning',
          'detect_gaps': 'Identify Knowledge Gaps',
          'recommend_new_docs': 'Recommend Documents'
        };
        
  const label = (actionLabels[actionId] || actionId).trim();
        let actionParams = null;
        
        // Set default parameters for certain actions
        if (actionId === 'set_top_k') {
          actionParams = { k: 15 }; // Default to 15 for "more depth"
        }
        
        if (actionId && actionLabels[actionId]) {
          let paramAttr = '';
          if (actionParams) {
            try {
              paramAttr = ` data-action-params="${esc(encodeURIComponent(JSON.stringify(actionParams)))}"`;
            } catch (e) {
              paramAttr = '';
            }
          }
          return `<button type="button" class="assistant-action secondary next-step" data-action-id="${actionId}" title="${esc(label)}"${paramAttr}>${esc(label)}</button>`;
        } else {
          // Fallback: treat as text suggestion
          let shortLabel = actionId;
          if (actionId.includes(':')) {
            shortLabel = actionId.split(':')[0].trim();
          } else if (actionId.length > 50) {
            shortLabel = actionId.substring(0, 47) + '...';
          }
          return `<button type="button" class="assistant-action secondary next-step" data-step-idx="${idx}" title="${esc(actionId)}">${esc(shortLabel)}</button>`;
        }
      }).join('');
      
  parts.push(`<div class="next-steps" style="margin-top:8px"><strong>Suggested actions:</strong><div style="margin-top:6px;display:flex;gap:8px;flex-wrap:wrap">${stepBtns}</div></div>`);
    }
    
      if (parts.length === 0) return;
     
    wrap.innerHTML = parts.join('');
    lastBot.appendChild(wrap);

    wrap.addEventListener('click', async (e) => {
      const btn = e.target.closest('button.assistant-action');
      if (!btn) return;
      e.preventDefault();
      
      const aid = btn.getAttribute('data-action-id');
      const stepIdx = btn.getAttribute('data-step-idx');
      
      if (aid) {
        // Define which actions should NOT send a user message immediately
        // Only upload_docs is a pure UI action now; all others regenerate
        const noImmediateMessageActions = new Set([
          'upload_docs'
        ]);
        
        if (!noImmediateMessageActions.has(aid)) {
          try {
            const chosenText = btn.textContent ? btn.textContent.trim() : aid;
            if (chosenText) {
              appendUserMessage(chosenText);
              // Persist immediately so it's not lost on refresh; mark as pending to complete later
              try { recordConversationEntry(chosenText, '', '', null); window.__pendingActionQuestion = chosenText; } catch {}
            }
          } catch {}
        }
        
        // Special handling for formula sheet - enable formula mode first
        if (aid === 'generate_formula_sheet') {
          const formulaCheckbox = document.getElementById('formula-mode');
          if (formulaCheckbox && !formulaCheckbox.checked) {
            formulaCheckbox.checked = true;
            highlightControl(formulaCheckbox.closest('.chat-settings-block'));
          }
        }
        
        // Parse optional params attached to the button
        let params = null;
        try {
          const raw = btn.getAttribute('data-action-params');
          if (raw) {
            try { params = JSON.parse(decodeURIComponent(raw)); } catch (e) { params = null; }
          }
        } catch (e) { params = null; }
        
        // Confirm if the action requests confirmation
        if (aid === 'upload_docs') {
          const ok = confirm('Open the upload dialog to attach documents?');
          if (!ok) return;
        }
        
        // Disable the button while performing
        const prevText = btn.textContent;
        btn.disabled = true;
        btn.textContent = 'Working…';
        try {
          await performAction(aid, params);
        } catch (err) {
          showError(err && err.message ? err.message : String(err));
        } finally {
          try { btn.disabled = false; btn.textContent = prevText; } catch {}
        }
      } else if (stepIdx !== null) {
        // This is a next-step suggestion button (fallback)
        const stepText = btn.textContent.trim();
        if (stepText) {
          // Fill the question input with the suggestion
          const questionInput = document.getElementById('question');
          if (questionInput) {
            questionInput.value = stepText;
            questionInput.focus();
            // Auto-resize the textarea
            questionInput.style.height = 'auto';
            questionInput.style.height = questionInput.scrollHeight + 'px';
          }
        }
      }
    });
  } catch (err) {
    console.error('[UI] maybeRenderActions encountered an error:', err);
    console.error('[UI] Error stack:', err.stack);
    try { console.warn('maybeRenderActions failed', err); } catch {}
  }
}

// Helper function to temporarily highlight a control
function highlightControl(element) {
  if (!element) return;
  element.style.transition = 'box-shadow 0.3s ease, background 0.3s ease';
  element.style.boxShadow = '0 0 20px rgba(107, 92, 255, 0.6)';
  element.style.background = 'rgba(107, 92, 255, 0.1)';
  setTimeout(() => {
    element.style.boxShadow = '';
    element.style.background = '';
  }, 2000);
}

// Add a PDF export button to a bot message containing formula sheet content
function addPdfExportButton(messageElement, markdown) {
  if (!messageElement) return;
  try {
    const exportDiv = document.createElement('div');
    exportDiv.style.marginTop = '12px';
    exportDiv.style.paddingTop = '8px';
    exportDiv.style.borderTop = '1px solid rgba(0, 0, 0, 0.06)';
    exportDiv.innerHTML = `
      <button class="assistant-action primary" style="font-size: 11px !important; padding: 6px 12px !important; height: auto !important;">
        📄 Export as PDF
      </button>
    `;
    const btn = exportDiv.querySelector('button');
    if (btn) {
      btn.addEventListener('click', () => {
        exportToPdf(messageElement, markdown);
      });
    }
    messageElement.appendChild(exportDiv);
  } catch (e) {
    console.error('Failed to add PDF export button:', e);
  }
}

// Export bot message content to PDF
function exportToPdf(messageElement, markdown) {
  try {
    // Use browser's print dialog with special styling for PDF
    const content = messageElement.cloneNode(true);
    
    // Remove the export button from the clone
    const exportBtn = content.querySelector('button');
    if (exportBtn) exportBtn.parentElement?.remove();
    
    // Create a temporary container for printing
    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      alert('Please allow popups to export PDF');
      return;
    }
    
    printWindow.document.write(`
      <!DOCTYPE html>
      <html>
      <head>
        <title>Formula Sheet - Anagnosis</title>
        <style>
          @page { margin: 1in; }
          body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            font-size: 12pt;
            line-height: 1.6;
            color: #1a1a1a;
            max-width: 100%;
            margin: 0;
            padding: 20px;
          }
          h1, h2, h3 { page-break-after: avoid; }
          table { page-break-inside: avoid; }
          pre, code {
            background: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
          }
          pre {
            padding: 12px;
            overflow-x: auto;
          }
          .next-steps, .trace, button { display: none !important; }
          img { max-width: 100%; height: auto; }
        </style>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js"></script>
      </head>
      <body>
        ${content.innerHTML}
      </body>
      </html>
    `);
    
    printWindow.document.close();
    
    // Wait for content to load, then trigger print
    printWindow.onload = () => {
      setTimeout(() => {
        printWindow.print();
      }, 500);
    };
  } catch (e) {
    console.error('Failed to export PDF:', e);
    alert('Failed to export PDF. Please try again.');
  }
}

// Perform an action by calling /api/ask/action and handling the result (UI actions or re-run answers)
async function performAction(action_id, params) {
  // Helper to complete last pending turn or create a new one
  const updateOrRecord = (q, ansHtml, ansMd) => {
    try {
      const last = chatHistory[chatHistory.length - 1];
      if (last && String(last.q || '') === String(q || '') && !(last.a_markdown || last.a || last.a_html)) {
        last.a_html = ansHtml || '';
        last.a_markdown = ansMd || (ansHtml ? stripHTML(ansHtml) : '');
        last.a = last.a_markdown;
        if (activeConvId && conversations[activeConvId]) {
          const conv = conversations[activeConvId];
          conv.history = [...chatHistory];
          conv.updatedAt = Date.now();
          conversations[activeConvId] = conv;
          saveConversations();
          renderConversationList();
        }
      } else {
        recordConversationEntry(q, ansHtml || '', ansMd || '');
      }
    } catch { recordConversationEntry(q, ansHtml || '', ansMd || ''); }
  };
  
  try {
    // Show immediate UI feedback: log + progress
    if (askLog) askLog.textContent = `Working… (${action_id})`;
    const sendBtn = askForm?.querySelector('.send-btn');
    if (sendBtn) sendBtn.disabled = true;
    if (askProgress) askProgress.classList.add('active');
    const question = (document.getElementById('question')?.value || '').trim() || (chatWindow?.querySelector('.message.user:last-of-type')?.textContent || '');
    const payload = collectAskPayloadFromUI();
    payload.action_id = action_id;
    if (params && typeof params === 'object') payload.action_params = params;
    payload.question = question;
    payload.history = chatHistory;
    const resp = await fetch(`${API_BASE}/ask/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      credentials: 'include',
    });
    if (!resp.ok) {
      const txt = await resp.text();
      if (askLog) askLog.textContent = `Error: ${txt || 'Action failed'}`;
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      throw new Error(txt || 'Action failed');
    }
    const data = await resp.json();
  // Record most recent action id for variety filtering
  try { window.__recentActionId = action_id; } catch {}
    // If the server returns a job id, switch to polling flow just like ask
    if (data && data.job_id) {
      const jobId = data.job_id;
      try { localStorage.setItem('anag_active_job', jobId); } catch {}
      const sendBtn = askForm?.querySelector('.send-btn');
      
      // Set up cancel handler for action jobs
      let cancelArmed = false;
      if (sendBtn) { 
        sendBtn.disabled = false; 
        sendBtn.classList.add('is-cancel'); 
        sendBtn.title = 'Cancel'; 
        try { sendBtn.setAttribute('type','button'); } catch {} 
        
        const cancelActionOnce = async (e) => {
          if (cancelArmed) return; 
          cancelArmed = true;
          e?.preventDefault?.();
          try { 
            await fetch(`${API_BASE}/ask/cancel/${jobId}`, { method: 'POST', credentials: 'include' }); 
          } catch {}
          // Update UI
          try {
            if (askLog) {
              const prev = String(askLog.textContent || '').trim();
              askLog.textContent = prev ? (prev + '\nCancelled') : 'Cancelled';
              try { askLog.scrollTop = askLog.scrollHeight; } catch {}
            }
          } catch {}
          try { if (askProgress) askProgress.classList.remove('active'); } catch {}
          try { if (sendBtn) { sendBtn.classList.remove('is-cancel'); sendBtn.title = 'Send'; sendBtn.setAttribute('type', 'submit'); } } catch {}
          try { if (window.__activeActionPoll) { clearInterval(window.__activeActionPoll); window.__activeActionPoll = null; } } catch {}
          try { localStorage.removeItem('anag_active_job'); } catch {}
          try {
            // Persist the action as a turn with an empty answer
            const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
            if (qForSave) {
              recordConversationEntry(qForSave, '', '', null);
              try { window.__pendingActionQuestion = null; } catch {}
            }
          } catch {}
        };
        
        try { sendBtn.addEventListener('click', cancelActionOnce, { once: true }); } catch {}
      }
      
      // Inline poller (simplified from askQuestion)
      const pollActionJob = async () => {
        try {
          const s = await fetch(`${API_BASE}/ask/status/${jobId}`, { credentials: 'include' });
          if (!s.ok) { throw new Error(`Status ${s.status}`); }
          const st = await s.json();
          if (Array.isArray(st.logs) && askLog) { askLog.textContent = st.logs.join('\n'); try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
          if (st.status === 'done') {
            if (askProgress) askProgress.classList.remove('active');
            if (sendBtn) { sendBtn.disabled = false; sendBtn.classList.remove('is-cancel'); sendBtn.title = 'Send'; try { sendBtn.setAttribute('type','submit'); } catch {} }
            try { localStorage.removeItem('anag_active_job'); } catch {}
            try { if (window.__activeActionPoll) { clearInterval(window.__activeActionPoll); window.__activeActionPoll = null; } } catch {}
            
            // Special handling for followup_questions in job results
            if (st.followup_questions && Array.isArray(st.followup_questions) && st.followup_questions.length > 0) {
              const followups = st.followup_questions;
              const esc = escapeHTML;
              const followupHTML = `
                <div class="followup-container" style="display: flex; flex-direction: column; gap: 10px;">
                  <div style="font-weight: 600; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">💡 Suggested Follow-up Questions</div>
                  ${followups.map((q, idx) => `
                    <button type="button" 
                            class="followup-question-btn" 
                            data-question="${esc(q)}"
                            style="
                              text-align: left;
                              padding: 12px 16px;
                              border: 1px solid rgba(79, 70, 229, 0.25);
                              background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(129, 140, 248, 0.06));
                              color: #1e293b;
                              border-radius: 12px;
                              cursor: pointer;
                              transition: all 0.2s ease;
                              font-size: 0.92rem;
                              line-height: 1.5;
                              box-shadow: 0 2px 8px -4px rgba(79, 70, 229, 0.15);
                            "
                            onmouseover="this.style.background='linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(129, 140, 248, 0.12))'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 4px 12px -4px rgba(79, 70, 229, 0.25)';"
                            onmouseout="this.style.background='linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(129, 140, 248, 0.06))'; this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px -4px rgba(79, 70, 229, 0.15)';"
                            onclick="this.disabled=true; this.style.opacity='0.6'; document.getElementById('question').value = this.dataset.question; document.getElementById('ask-form').requestSubmit();">
                      <span style="display: inline-block; margin-right: 8px; opacity: 0.7;">${idx + 1}.</span>
                      ${esc(q)}
                    </button>
                  `).join('')}
                </div>
              `;
              const botMessageEl = appendBotMessage(followupHTML);
              try {
                const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
                updateOrRecord(qForSave, followupHTML, `Follow-up Questions:\n${followups.map((q, i) => `${i+1}. ${q}`).join('\n')}`);
                try { window.__pendingActionQuestion = null; } catch {}
              } catch {}
              return 'done';
            }
            
            // Special handling for quiz_questions in job results
            if (st.quiz_questions && Array.isArray(st.quiz_questions) && st.quiz_questions.length > 0) {
              const quizData = st.quiz_questions;
              const esc = escapeHTML;
              
                // Define the quiz handler function globally before rendering HTML
                window.quizState = window.quizState || {};
                window.quizState.answered = 0;
                window.quizState.correct = 0;
                window.quizState.total = quizData.length;
                // Regenerate helper
                window.regenQuiz = async function(count){
                  try {
                    let n = Number(count);
                    if (!Number.isFinite(n)) n = quizData.length || 5;
                    n = Math.max(1, Math.min(30, Math.round(n)));
                    // Optional: show a small user message for clarity
                    appendUserMessage(`Regenerate quiz with ${n} question${n===1?'':'s'}`);
                    try { recordConversationEntry(`Regenerate quiz (${n})`, '', '', null); } catch {}
                    await performAction('generate_quiz', { count: n });
                  } catch (e) {
                    showError(e && e.message ? e.message : String(e));
                  }
                };
              
                window.handleQuizAnswer = function(btn, qIdx, optIdx, correctIdx) {
                  // Get explanation from data attribute
                  const explanation = btn.getAttribute('data-explanation') || '';
                
                  const container = btn.closest('.quiz-question');
                  const wrap = btn.closest('.quiz-container');
                  const allBtns = container.querySelectorAll('.quiz-option-btn');
                  allBtns.forEach(b => b.disabled = true);
                
                  const isCorrect = (optIdx === correctIdx);
                
                  if (isCorrect) {
                    btn.style.background = 'rgba(34, 197, 94, 0.15)';
                    btn.style.borderColor = 'rgba(34, 197, 94, 0.5)';
                    btn.style.color = '#166534';
                    btn.classList.add('correct');
                    window.quizState.correct++;
                  } else {
                    btn.style.background = 'rgba(239, 68, 68, 0.15)';
                    btn.style.borderColor = 'rgba(239, 68, 68, 0.5)';
                    btn.style.color = '#991b1b';
                    btn.classList.add('incorrect');
                  
                    allBtns[correctIdx].style.background = 'rgba(34, 197, 94, 0.15)';
                    allBtns[correctIdx].style.borderColor = 'rgba(34, 197, 94, 0.5)';
                    allBtns[correctIdx].style.color = '#166534';
                    allBtns[correctIdx].classList.add('correct');
                  }
                
                  const feedback = container.querySelector('.quiz-feedback');
                  feedback.style.display = 'block';
                  if (isCorrect) {
                    feedback.style.background = 'rgba(34, 197, 94, 0.12)';
                    feedback.style.border = '1px solid rgba(34, 197, 94, 0.3)';
                    feedback.style.color = '#166534';
                    feedback.innerHTML = '<strong>✓ Correct!</strong> ' + (explanation || '');
                  } else {
                    feedback.style.background = 'rgba(239, 68, 68, 0.12)';
                    feedback.style.border = '1px solid rgba(239, 68, 68, 0.3)';
                    feedback.style.color = '#991b1b';
                    feedback.innerHTML = '<strong>✗ Incorrect.</strong> The correct answer is <strong>' + String.fromCharCode(65 + correctIdx) + '</strong>. ' + (explanation || '');
                  }
                
                  window.quizState.answered++;
                
                  // Reveal next question sequentially
                  try {
                    const next = wrap?.querySelector(`.quiz-question[data-quiz-idx='${qIdx+1}']`);
                    if (next && next.style.display === 'none') {
                      next.style.display = 'block';
                      next.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                  } catch {}

                  if (window.quizState.answered === window.quizState.total) {
                    const scoreEl = wrap ? wrap.querySelector('.quiz-score') : document.querySelector('.quiz-score');
                    const percentage = Math.round((window.quizState.correct / window.quizState.total) * 100);
                    let emoji = '🎉';
                    if (percentage < 50) emoji = '📚';
                    else if (percentage < 80) emoji = '👍';
                  
                    scoreEl.innerHTML = emoji + ' You scored <strong>' + window.quizState.correct + '/' + window.quizState.total + '</strong> (' + percentage + '%)';
                    scoreEl.style.display = 'block';
                  }
                };
              
              const initialCount = (st.quiz_meta && Number(st.quiz_meta.count)) || quizData.length || 5;
              const quizHTML = `
                <div class="quiz-container" style="display: flex; flex-direction: column; gap: 16px;">
                  <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">📝 Quiz Time! <span style="font-weight:600; color:#475569">(${initialCount} questions)</span></div>
                  <div style="font-size:.85rem; color:#64748b; margin-bottom:8px;">Questions unlock one-by-one as you answer.</div>
                  ${quizData.map((q, qIdx) => `
                    <div class="quiz-question" data-quiz-idx="${qIdx}" style="${qIdx>0?'display:none;':''}padding: 16px; border: 1px solid rgba(79, 70, 229, 0.2); background: rgba(249, 250, 251, 0.8); border-radius: 12px;">
                      <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; color: #1e293b;">
                        <span style="display: inline-block; width: 28px; height: 28px; border-radius: 50%; background: rgba(79, 70, 229, 0.15); text-align: center; line-height: 28px; margin-right: 8px; font-weight: 700; color: #4f46e5;">${qIdx + 1}</span>
                        ${esc(q.question || '')}
                      </div>
                      <div class="quiz-options" style="display: flex; flex-direction: column; gap: 8px;">
                        ${(q.options || []).map((opt, optIdx) => {
                          const escapedExplanation = esc(q.explanation || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                          return `
                          <button type="button" 
                                  class="quiz-option-btn" 
                                  data-quiz-idx="${qIdx}"
                                  data-option-idx="${optIdx}"
                                  data-correct="${q.correct || 0}"
                                  data-explanation="${escapedExplanation}"
                                  style="
                                    text-align: left;
                                    padding: 10px 14px;
                                    border: 1.5px solid rgba(148, 163, 184, 0.3);
                                    background: #ffffff;
                                    color: #1e293b;
                                    border-radius: 10px;
                                    cursor: pointer;
                                    transition: all 0.2s ease;
                                    font-size: 0.9rem;
                                    position: relative;
                                  "
                                  onmouseover="if(!this.disabled) { this.style.background='rgba(249, 250, 251, 1)'; this.style.borderColor='rgba(79, 70, 229, 0.4)'; }"
                                  onmouseout="if(!this.disabled && !this.classList.contains('correct') && !this.classList.contains('incorrect')) { this.style.background='#ffffff'; this.style.borderColor='rgba(148, 163, 184, 0.3)'; }"
                                  onclick="handleQuizAnswer(this, ${qIdx}, ${optIdx}, ${q.correct || 0})">
                            <span style="display: inline-block; width: 24px; height: 24px; border-radius: 50%; border: 2px solid currentColor; text-align: center; line-height: 20px; margin-right: 10px; font-weight: 600;">${String.fromCharCode(65 + optIdx)}</span>
                            ${esc(opt)}
                          </button>
                        `;
                        }).join('')}
                      </div>
                      <div class="quiz-feedback" style="display: none; margin-top: 12px; padding: 12px; border-radius: 8px; font-size: 0.88rem; line-height: 1.5;"></div>
                    </div>
                  `).join('')}
                  <div class="quiz-score" style="display: none; padding: 16px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(59, 130, 246, 0.12)); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; text-align: center; font-weight: 600; font-size: 1rem;"></div>
                  <div class="quiz-regenerate" style="display:flex; align-items:center; gap:10px; justify-content:flex-end; margin-top:4px;">
                    <label style="display:inline-flex; align-items:center; gap:6px; font-size:.9rem; color:#334155;">
                      <span>Questions:</span>
                      <input type="number" class="quiz-count" min="1" max="30" value="${initialCount}" style="width:64px; text-align:center; border:1px solid rgba(148,163,184,0.4); border-radius:8px; padding:4px;" onchange="if(this.value>30)this.value=30;if(this.value<1)this.value=1;" onblur="if(this.value>30)this.value=30;if(this.value<1)this.value=1;" />
                    </label>
                    <button type="button" class="assistant-action secondary" onclick="(function(el){ const wrap = el.closest('.quiz-container'); const inp = wrap ? wrap.querySelector('.quiz-count') : null; const n = inp ? inp.value : ${initialCount}; window.regenQuiz(n); })(this)" title="Generate a new quiz with this many questions">Regenerate quiz</button>
                  </div>
                </div>
              `;
              const botMessageEl = appendBotMessage(quizHTML);
              try {
                const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
                const quizMd = 'Quiz:\n' + quizData.map((q, i) => `${i+1}. ${q.question}\n${(q.options || []).map((o, j) => `   ${String.fromCharCode(65+j)}) ${o}`).join('\n')}\nAnswer: ${String.fromCharCode(65 + (q.correct || 0))}`).join('\n\n');
                updateOrRecord(qForSave, quizHTML, quizMd);
                try { window.__pendingActionQuestion = null; } catch {}
              } catch {}
              return 'done';
            }
            
            if (st.answer) {
              // Render markdown to HTML before displaying
              const answerHtml = renderMarkdownToHtml(st.answer_markdown || st.answer);
              const botMessageEl = appendBotMessage(answerHtml);
              try {
                const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
                updateOrRecord(qForSave, answerHtml || '', st.answer_markdown || st.answer || '');
                try { window.__pendingActionQuestion = null; } catch {}
              } catch {}
              try { maybeRenderActions(st.actions || [], st.next_steps || [], botMessageEl); } catch {}
              try { 
                if (st.trace || st.result?.trace) { 
                  maybeAppendTrace(st.trace || st.result?.trace, botMessageEl); 
                } else {
                  maybeAppendTrace(null, botMessageEl);
                }
              } catch {}
              // Add PDF export button if this was a formula sheet action
              if (action_id === 'generate_formula_sheet') {
                addPdfExportButton(botMessageEl, st.answer_markdown || st.answer || '');
              }
            }
            return 'done';
          }
          return st.status || 'running';
        } catch (e) {
          if (askLog) askLog.textContent = `Error: ${e}`;
          if (askProgress) askProgress.classList.remove('active');
          if (sendBtn) sendBtn.disabled = false;
          try { localStorage.removeItem('anag_active_job'); } catch {}
          return 'error';
        }
      };
      // kick once and then interval
      let status = await pollActionJob();
      if (status !== 'done' && status !== 'error') {
        const int = setInterval(async () => {
          const st = await pollActionJob();
          if (st === 'done' || st === 'error') clearInterval(int);
        }, 900);
        // Store interval handle for cancellation
        window.__activeActionPoll = int;
      }
      return;
    }
    try {
      // Write any logs returned by the server into the Ask Log
      if (Array.isArray(data.logs) && data.logs.length && askLog) {
        askLog.textContent = data.logs.join('\n');
        try { askLog.scrollTop = askLog.scrollHeight; } catch {}
      } else if (data.log && askLog) {
        askLog.textContent = String(data.log);
        try { askLog.scrollTop = askLog.scrollHeight; } catch {}
      }
    } catch {}
    
    // UI hints
    if (data.ui && data.ui.open_upload) {
      // Switch to library tab and focus the upload input
      try { switchTab('library'); } catch {}
      try { const inp = document.getElementById('upload-input'); if (inp) inp.click(); } catch {}
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    }
    
    // Special handling for followup_questions: render as clickable buttons
    if (data.followup_questions && Array.isArray(data.followup_questions) && data.followup_questions.length > 0) {
      const followups = data.followup_questions;
      const esc = escapeHTML;
      // Create a custom message with interactive buttons
      const followupHTML = `
        <div class="followup-container" style="display: flex; flex-direction: column; gap: 10px;">
          <div style="font-weight: 600; font-size: 1rem; color: #0f172a; margin-bottom: 4px;">💡 Suggested Follow-up Questions</div>
          ${followups.map((q, idx) => `
            <button type="button" 
                    class="followup-question-btn" 
                    data-question="${esc(q)}"
                    style="
                      text-align: left;
                      padding: 12px 16px;
                      border: 1px solid rgba(79, 70, 229, 0.25);
                      background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(129, 140, 248, 0.06));
                      color: #1e293b;
                      border-radius: 12px;
                      cursor: pointer;
                      transition: all 0.2s ease;
                      font-size: 0.92rem;
                      line-height: 1.5;
                      box-shadow: 0 2px 8px -4px rgba(79, 70, 229, 0.15);
                    "
                    onmouseover="this.style.background='linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(129, 140, 248, 0.12))'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 4px 12px -4px rgba(79, 70, 229, 0.25)';"
                    onmouseout="this.style.background='linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(129, 140, 248, 0.06))'; this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 8px -4px rgba(79, 70, 229, 0.15)';"
                    onclick="this.disabled=true; this.style.opacity='0.6'; document.getElementById('question').value = this.dataset.question; document.getElementById('ask-form').requestSubmit();">
              <span style="display: inline-block; margin-right: 8px; opacity: 0.7;">${idx + 1}.</span>
              ${esc(q)}
            </button>
          `).join('')}
        </div>
      `;
      var botMessageEl = appendBotMessage(followupHTML);
      try {
        const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
        updateOrRecord(qForSave, followupHTML, `Follow-up Questions:\n${followups.map((q, i) => `${i+1}. ${q}`).join('\n')}`);
        try { window.__pendingActionQuestion = null; } catch {}
      } catch {}
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    }
    
    // Special handling for quiz_questions: render as interactive quiz
    if (data.quiz_questions && Array.isArray(data.quiz_questions) && data.quiz_questions.length > 0) {
      const quizData = data.quiz_questions;
      const esc = escapeHTML;
      
        // Define the quiz handler function globally before rendering HTML
        window.quizState = window.quizState || {};
        window.quizState.answered = 0;
        window.quizState.correct = 0;
        window.quizState.total = quizData.length;
        // Regenerate helper
        window.regenQuiz = async function(count){
          try {
            let n = Number(count);
            if (!Number.isFinite(n)) n = quizData.length || 5;
            n = Math.max(1, Math.min(30, Math.round(n)));
            appendUserMessage(`Regenerate quiz with ${n} question${n===1?'':'s'}`);
            try { recordConversationEntry(`Regenerate quiz (${n})`, '', '', null); } catch {}
            await performAction('generate_quiz', { count: n });
          } catch (e) {
            showError(e && e.message ? e.message : String(e));
          }
        };
      
        window.handleQuizAnswer = function(btn, qIdx, optIdx, correctIdx) {
          // Get explanation from data attribute
          const explanation = btn.getAttribute('data-explanation') || '';
        
          // Prevent multiple answers
          const container = btn.closest('.quiz-question');
          const wrap = btn.closest('.quiz-container');
          const allBtns = container.querySelectorAll('.quiz-option-btn');
          allBtns.forEach(b => b.disabled = true);
        
          const isCorrect = (optIdx === correctIdx);
        
          // Style the clicked button
          if (isCorrect) {
            btn.style.background = 'rgba(34, 197, 94, 0.15)';
            btn.style.borderColor = 'rgba(34, 197, 94, 0.5)';
            btn.style.color = '#166534';
            btn.classList.add('correct');
            window.quizState.correct++;
          } else {
            btn.style.background = 'rgba(239, 68, 68, 0.15)';
            btn.style.borderColor = 'rgba(239, 68, 68, 0.5)';
            btn.style.color = '#991b1b';
            btn.classList.add('incorrect');
          
            // Highlight correct answer
            allBtns[correctIdx].style.background = 'rgba(34, 197, 94, 0.15)';
            allBtns[correctIdx].style.borderColor = 'rgba(34, 197, 94, 0.5)';
            allBtns[correctIdx].style.color = '#166534';
            allBtns[correctIdx].classList.add('correct');
          }
        
          // Show feedback
          const feedback = container.querySelector('.quiz-feedback');
          feedback.style.display = 'block';
          if (isCorrect) {
            feedback.style.background = 'rgba(34, 197, 94, 0.12)';
            feedback.style.border = '1px solid rgba(34, 197, 94, 0.3)';
            feedback.style.color = '#166534';
            feedback.innerHTML = '<strong>✓ Correct!</strong> ' + (explanation || '');
          } else {
            feedback.style.background = 'rgba(239, 68, 68, 0.12)';
            feedback.style.border = '1px solid rgba(239, 68, 68, 0.3)';
            feedback.style.color = '#991b1b';
            feedback.innerHTML = '<strong>✗ Incorrect.</strong> The correct answer is <strong>' + String.fromCharCode(65 + correctIdx) + '</strong>. ' + (explanation || '');
          }
        
          window.quizState.answered++;
        
          // Show final score when all questions answered
          // Reveal next question sequentially
          try {
            const next = wrap?.querySelector(`.quiz-question[data-quiz-idx='${qIdx+1}']`);
            if (next && next.style.display === 'none') {
              next.style.display = 'block';
              next.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
          } catch {}

          if (window.quizState.answered === window.quizState.total) {
            const scoreEl = wrap ? wrap.querySelector('.quiz-score') : document.querySelector('.quiz-score');
            const percentage = Math.round((window.quizState.correct / window.quizState.total) * 100);
            let emoji = '🎉';
            if (percentage < 50) emoji = '📚';
            else if (percentage < 80) emoji = '👍';
          
            scoreEl.innerHTML = emoji + ' You scored <strong>' + window.quizState.correct + '/' + window.quizState.total + '</strong> (' + percentage + '%)';
            scoreEl.style.display = 'block';
          }
        };
      
        const initialCount = (data.quiz_meta && Number(data.quiz_meta.count)) || quizData.length || 5;
        const quizHTML = `
        <div class="quiz-container" style="display: flex; flex-direction: column; gap: 16px;">
          <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">📝 Quiz Time! <span style="font-weight:600; color:#475569">(${initialCount} questions)</span></div>
          <div style="font-size:.85rem; color:#64748b; margin-bottom:8px;">Questions unlock one-by-one as you answer.</div>
          ${quizData.map((q, qIdx) => `
            <div class="quiz-question" data-quiz-idx="${qIdx}" style="${qIdx>0?'display:none;':''}padding: 16px; border: 1px solid rgba(79, 70, 229, 0.2); background: rgba(249, 250, 251, 0.8); border-radius: 12px;">
              <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 12px; color: #1e293b;">
                <span style="display: inline-block; width: 28px; height: 28px; border-radius: 50%; background: rgba(79, 70, 229, 0.15); text-align: center; line-height: 28px; margin-right: 8px; font-weight: 700; color: #4f46e5;">${qIdx + 1}</span>
                ${esc(q.question || '')}
              </div>
              <div class="quiz-options" style="display: flex; flex-direction: column; gap: 8px;">
                ${(q.options || []).map((opt, optIdx) => {
                  const escapedExplanation = esc(q.explanation || '').replace(/'/g, "\\'").replace(/"/g, '&quot;');
                  return `
                  <button type="button" 
                          class="quiz-option-btn" 
                          data-quiz-idx="${qIdx}"
                          data-option-idx="${optIdx}"
                          data-correct="${q.correct || 0}"
                          data-explanation="${escapedExplanation}"
                          style="
                            text-align: left;
                            padding: 10px 14px;
                            border: 1.5px solid rgba(148, 163, 184, 0.3);
                            background: #ffffff;
                            color: #1e293b;
                            border-radius: 10px;
                            cursor: pointer;
                            transition: all 0.2s ease;
                            font-size: 0.9rem;
                            position: relative;
                          "
                          onmouseover="if(!this.disabled) { this.style.background='rgba(249, 250, 251, 1)'; this.style.borderColor='rgba(79, 70, 229, 0.4)'; }"
                          onmouseout="if(!this.disabled && !this.classList.contains('correct') && !this.classList.contains('incorrect')) { this.style.background='#ffffff'; this.style.borderColor='rgba(148, 163, 184, 0.3)'; }"
                          onclick="handleQuizAnswer(this, ${qIdx}, ${optIdx}, ${q.correct || 0})">
                    <span style="display: inline-block; width: 24px; height: 24px; border-radius: 50%; border: 2px solid currentColor; text-align: center; line-height: 20px; margin-right: 10px; font-weight: 600;">${String.fromCharCode(65 + optIdx)}</span>
                    ${esc(opt)}
                  </button>
                `;
                }).join('')}
              </div>
              <div class="quiz-feedback" style="display: none; margin-top: 12px; padding: 12px; border-radius: 8px; font-size: 0.88rem; line-height: 1.5;"></div>
            </div>
          `).join('')}
          <div class="quiz-score" style="display: none; padding: 16px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.12), rgba(59, 130, 246, 0.12)); border: 1px solid rgba(34, 197, 94, 0.3); border-radius: 12px; text-align: center; font-weight: 600; font-size: 1rem;"></div>
          <div class="quiz-regenerate" style="display:flex; align-items:center; gap:10px; justify-content:flex-end; margin-top:4px;">
            <label style="display:inline-flex; align-items:center; gap:6px; font-size:.9rem; color:#334155;">
              <span>Questions:</span>
              <input type="number" class="quiz-count" min="1" max="30" value="${initialCount}" style="width:64px; text-align:center; border:1px solid rgba(148,163,184,0.4); border-radius:8px; padding:4px;" onchange="if(this.value>30)this.value=30;if(this.value<1)this.value=1;" onblur="if(this.value>30)this.value=30;if(this.value<1)this.value=1;" />
            </label>
            <button type="button" class="assistant-action secondary" onclick="(function(el){ const wrap = el.closest('.quiz-container'); const inp = wrap ? wrap.querySelector('.quiz-count') : null; const n = inp ? inp.value : ${initialCount}; window.regenQuiz(n); })(this)" title="Generate a new quiz with this many questions">Regenerate quiz</button>
          </div>
        </div>
      `;
      var botMessageEl = appendBotMessage(quizHTML);
      try {
        const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
        const quizMd = 'Quiz:\n' + quizData.map((q, i) => `${i+1}. ${q.question}\n${(q.options || []).map((o, j) => `   ${String.fromCharCode(65+j)}) ${o}`).join('\n')}\nAnswer: ${String.fromCharCode(65 + (q.correct || 0))}`).join('\n\n');
        updateOrRecord(qForSave, quizHTML, quizMd);
        try { window.__pendingActionQuestion = null; } catch {}
      } catch {}
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    }
    
    // If server returned a new answer, append it
    if (data.answer) {
      // Render markdown to HTML before displaying
      const answerHtml = renderMarkdownToHtml(data.answer_markdown || data.answer);
      var botMessageEl = appendBotMessage(answerHtml);
      try {
        const qForSave = (window.__pendingActionQuestion || payload.question || question || '').trim();
        updateOrRecord(qForSave, answerHtml || '', data.answer_markdown || data.answer || '');
        try { window.__pendingActionQuestion = null; } catch {}
      } catch {}
      // Add PDF export button if this was a formula sheet action
      if (action_id === 'generate_formula_sheet') {
        addPdfExportButton(botMessageEl, data.answer_markdown || data.answer || '');
      }
    }
    // If server returned a result with actions, render them
    const result = data.result || {};
    try {
      const actions = Array.isArray(result.actions) ? result.actions : (Array.isArray(data.actions) ? data.actions : []);
      let next_steps = Array.isArray(result.next_steps) ? result.next_steps : (Array.isArray(data.next_steps) ? data.next_steps : []);
      // Fallback: ensure next steps always exist (client-side defaults)
      if (!Array.isArray(next_steps) || next_steps.length === 0) {
        next_steps = ['expand_detail', 'upload_docs', 'generate_formula_sheet', 'broaden_docs'];
      }
      // Prefer rendering on the freshly appended bot message
      maybeRenderActions(actions || [], next_steps || [], botMessageEl);
      // Attach trace to this new message if present
      try {
        if (data.trace || result.trace) { 
          maybeAppendTrace(data.trace || result.trace, botMessageEl); 
        } else { 
          maybeAppendTrace(null, botMessageEl); 
        }
      } catch {}
    } catch {}
    // Turn off progress and re-enable send
    try { if (sendBtn) sendBtn.disabled = false; } catch {}
    try { if (askProgress) askProgress.classList.remove('active'); } catch {}
  } catch (err) {
    try {
      if (askLog) askLog.textContent = `Error: ${err && err.message ? err.message : String(err)}`;
      if (askProgress) askProgress.classList.remove('active');
      const sendBtn = askForm?.querySelector('.send-btn');
      if (sendBtn) sendBtn.disabled = false;
    } catch {}
    throw err;
  }
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
        // Log the response for debugging
        try { console.log('[UI] Direct ask response:', d); } catch {}
        try { console.log('[UI] Actions received:', d.actions); } catch {}
        try { console.log('[UI] Next steps received:', d.next_steps); } catch {}
        if (d.answer) {
          // Render markdown to HTML before displaying
          const answerHtml = renderMarkdownToHtml(d.answer_markdown || d.answer);
          const botMessageEl = appendBotMessage(answerHtml);
          // Wait for DOM update before attaching actions
          setTimeout(() => {
            try { 
              maybeRenderActions(d.actions || [], d.next_steps || [], botMessageEl); 
            } catch (e) {
              console.error('[UI] Error rendering actions/next_steps:', e);
            }
          }, 100);
          try {
            if (d.trace) { maybeAppendTrace(d.trace, botMessageEl); }
            else { maybeAppendTrace(null, botMessageEl); }
          } catch {}
        }
        // Trace moved to the freshly appended bot message above
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
      try { if (window.__activeActionPoll) { clearInterval(window.__activeActionPoll); window.__activeActionPoll = null; } } catch {}
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
                // Render markdown to HTML before displaying
                const answerHtml = renderMarkdownToHtml(d.answer_markdown || d.answer);
                appendBotMessage(answerHtml);
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
        // Log the full response for debugging
        try { console.log('[UI] Ask status response:', st); } catch {}
        try { console.log('[UI] Actions received:', st.actions); } catch {}
        try { console.log('[UI] Next steps received:', st.next_steps); } catch {}
        try { console.log('[UI] Next steps is array?', Array.isArray(st.next_steps), 'length:', st.next_steps?.length); } catch {}
        if (st.answer) {
          // Render markdown to HTML before displaying
          const answerHtml = renderMarkdownToHtml(st.answer_markdown || st.answer);
          const botMessageEl = appendBotMessage(answerHtml);
          try { 
            // Actions and next_steps are at the top level of st, not nested in result
            const actionsToRender = st.actions || [];
            const nextStepsToRender = st.next_steps || [];
            maybeRenderActions(actionsToRender, nextStepsToRender, botMessageEl);
          } catch (e) {
            console.error('[UI] Error rendering actions/next_steps:', e);
          }
          try {
            if (st.trace) { maybeAppendTrace(st.trace, botMessageEl); }
            else { maybeAppendTrace(null, botMessageEl); }
          } catch {}
        }
        // Trace moved to the freshly appended bot message above
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
    serp_key: getEl('serp-key')?.value || '',
    brave_key: getEl('brave-key')?.value || '',
    embed_backend: getEl('embed-backend')?.value || 'hf',
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
    // Disable controls that require OpenAI if key is missing (do not hide)
    try {
      const ok = data && data.keys && data.keys.openai;
      const openaiPresent = !!(ok && ok.present);
      const webToggle = document.getElementById('web-enabled');
      if (webToggle) {
        webToggle.disabled = !openaiPresent;
        webToggle.title = openaiPresent ? '' : 'Requires OpenAI API key';
      }
      const providerSelect = document.getElementById('web-provider');
      if (providerSelect) {
        providerSelect.disabled = !openaiPresent;
        providerSelect.title = openaiPresent ? '' : 'Requires OpenAI API key';
      }
    } catch {}
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
  
  // Render actions on the last bot message if one exists
  if (lastEl && lastEl.classList && lastEl.classList.contains('bot')) {
    try {
      // Provide fallback next steps for the last message
      const defaultNextSteps = ['expand_detail', 'upload_docs', 'generate_formula_sheet', 'broaden_docs'];
      maybeRenderActions([], defaultNextSteps, lastEl);
    } catch (e) {
      console.error('[UI] Error rendering actions on loaded conversation:', e);
    }
  }
  
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
  // Check if user is a dev by checking if dev panel exists
  const devSection = document.getElementById('dev');
  if (!devSection) {
    // Not a dev user, skip all dev tools initialization
    return;
  }
  
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
  
  // Create verified user
  document.getElementById('dev-create-user')?.addEventListener('click', async () => {
    const emailEl = document.getElementById('dev-user-email');
    const passEl = document.getElementById('dev-user-password');
    const resultEl = document.getElementById('dev-user-result');
    
    const email = emailEl?.value?.trim();
    const password = passEl?.value;
    
    if (!email || !password) {
      if (resultEl) {
        resultEl.textContent = 'Email and password required';
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/admin/create_verified_user?email=${encodeURIComponent(email)}&password=${encodeURIComponent(password)}`, {
        method: 'POST',
        credentials: 'include'
      });
      
      const data = await response.json();
      
      if (response.ok) {
        if (resultEl) {
          resultEl.textContent = `✓ User created: ${data.user.email} (verified)`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#efe';
          resultEl.style.color = '#060';
        }
        if (emailEl) emailEl.value = '';
        if (passEl) passEl.value = '';
      } else {
        if (resultEl) {
          resultEl.textContent = `Error: ${data.detail || 'Failed to create user'}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#fee';
          resultEl.style.color = '#c00';
        }
      }
    } catch (err) {
      if (resultEl) {
        resultEl.textContent = `Network error: ${err}`;
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
    }
  });
  
  // Delete user
  document.getElementById('dev-delete-user')?.addEventListener('click', async () => {
    const emailEl = document.getElementById('dev-manage-email');
    const resultEl = document.getElementById('dev-manage-result');
    const email = emailEl?.value?.trim();
    
    if (!email) {
      if (resultEl) {
        resultEl.textContent = 'Email required';
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
      return;
    }
    
    if (!confirm(`Delete user ${email} and all their data?`)) return;
    
    try {
      const response = await fetch(`${API_BASE}/admin/delete_user?email=${encodeURIComponent(email)}`, {
        method: 'POST',
        credentials: 'include'
      });
      const data = await response.json();
      if (response.ok) {
        if (resultEl) {
          resultEl.textContent = `✓ ${data.message}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#efe';
          resultEl.style.color = '#060';
        }
        if (emailEl) emailEl.value = '';
      } else {
        if (resultEl) {
          resultEl.textContent = `Error: ${data.detail || 'Failed'}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#fee';
          resultEl.style.color = '#c00';
        }
      }
    } catch (err) {
      if (resultEl) {
        resultEl.textContent = `Network error: ${err}`;
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
    }
  });
  
  // Ban email
  document.getElementById('dev-ban-email')?.addEventListener('click', async () => {
    const emailEl = document.getElementById('dev-manage-email');
    const resultEl = document.getElementById('dev-manage-result');
    const email = emailEl?.value?.trim();
    
    if (!email) {
      if (resultEl) {
        resultEl.textContent = 'Email required';
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
      return;
    }
    
    if (!confirm(`Ban ${email} from signing up?`)) return;
    
    try {
      const response = await fetch(`${API_BASE}/admin/ban_email?email=${encodeURIComponent(email)}`, {
        method: 'POST',
        credentials: 'include'
      });
      const data = await response.json();
      if (response.ok) {
        if (resultEl) {
          resultEl.textContent = `✓ ${data.message}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#efe';
          resultEl.style.color = '#060';
        }
        if (emailEl) emailEl.value = '';
        loadBannedEmails();
      } else {
        if (resultEl) {
          resultEl.textContent = `Error: ${data.detail || 'Failed'}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#fee';
          resultEl.style.color = '#c00';
        }
      }
    } catch (err) {
      if (resultEl) {
        resultEl.textContent = `Network error: ${err}`;
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
    }
  });
  
  // Unban email
  document.getElementById('dev-unban-email')?.addEventListener('click', async () => {
    const emailEl = document.getElementById('dev-manage-email');
    const resultEl = document.getElementById('dev-manage-result');
    const email = emailEl?.value?.trim();
    
    if (!email) {
      if (resultEl) {
        resultEl.textContent = 'Email required';
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
      return;
    }
    
    try {
      const response = await fetch(`${API_BASE}/admin/unban_email?email=${encodeURIComponent(email)}`, {
        method: 'POST',
        credentials: 'include'
      });
      const data = await response.json();
      if (response.ok) {
        if (resultEl) {
          resultEl.textContent = `✓ ${data.message}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#efe';
          resultEl.style.color = '#060';
        }
        if (emailEl) emailEl.value = '';
        loadBannedEmails();
      } else {
        if (resultEl) {
          resultEl.textContent = `Error: ${data.detail || 'Failed'}`;
          resultEl.style.display = 'block';
          resultEl.style.background = '#fee';
          resultEl.style.color = '#c00';
        }
      }
    } catch (err) {
      if (resultEl) {
        resultEl.textContent = `Network error: ${err}`;
        resultEl.style.display = 'block';
        resultEl.style.background = '#fee';
        resultEl.style.color = '#c00';
      }
    }
  });
  
  // Load banned emails
  async function loadBannedEmails() {
    const listEl = document.getElementById('dev-banned-list');
    if (!listEl) return;
    try {
      const response = await fetch(`${API_BASE}/admin/banned_emails`, { credentials: 'include' });
      const data = await response.json();
      if (data.banned && data.banned.length > 0) {
        listEl.textContent = data.banned.join('\n');
      } else {
        listEl.textContent = '(no banned emails)';
      }
    } catch {
      listEl.textContent = '(error loading)';
    }
  }
  loadBannedEmails();
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
