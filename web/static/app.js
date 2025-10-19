const tabs = document.querySelectorAll('.anag-tabs button');
try { console.info('[anag-ui] app.js v35 loaded'); } catch {}
const panels = document.querySelectorAll('.tab-panel');
const uploadForm = document.getElementById('upload-form');
const ingestLog = document.getElementById('ingest-log');
const askForm = document.getElementById('ask-form');
const askLog = document.getElementById('ask-log');
const askProgress = document.getElementById('ask-progress');
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
const refreshLibraryBtn = document.getElementById('refresh-library'); // may not exist
const onlyDocSelect = document.getElementById('only-doc');
const saveLibraryBtn = document.getElementById('save-library');
const clearLibraryBtn = document.getElementById('clear-library');
const cancelLibraryBtn = document.getElementById('cancel-library');
const settingsForm = document.getElementById('settings-form');
const settingsStatus = document.getElementById('settings-status');
const uploadProgressTrack = document.getElementById('upload-progress-track');
const uploadProgressBar = document.getElementById('upload-progress');
const uploadProgressWrap = uploadProgressTrack ? uploadProgressTrack.parentElement : null;
const bootstrapEl = document.getElementById('bootstrap-settings');
const uploadListEl = document.getElementById('upload-list');
let selectedFiles = [];
const dropZone = document.querySelector('.ingest-dropzone');

const API_BASE = '/api';
const AUTH_BASE = 'http://localhost:8000/api/auth';
const chatHistory = [];
const STORAGE_KEY = 'anag_conversations_v1';
let conversations = {};
let activeConvId = null;
const ingestPolls = new Map();
let defaultsData = {};
const pendingRemovals = new Set();
let librarySaveController = null;

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
  if (rerankerSettings) rerankerSettings.value = defaults.ASK_RERANKER || 'off';
  setValue('ask-candidates', defaults.ASK_CANDIDATES);
}

applyDefaults(defaultsData);

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

function renderLibraryBooks(items) {
  if (!libraryGrid || !libraryEmpty) return;
  if (!Array.isArray(items) || !items.length) {
    libraryGrid.innerHTML = '';
    libraryGrid.classList.add('empty');
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
}

async function refreshLibrary() {
  if (!libraryGrid) return;
  libraryGrid.classList.add('loading');
  if (libraryEmpty) libraryEmpty.style.display = 'block';
  libraryGrid.innerHTML = '';
  try {
    const resp = await fetch(`${API_BASE}/library`, { credentials: 'include' });
    if (!resp.ok) throw new Error(await resp.text());
    const data = await resp.json();
    renderLibraryBooks(data.documents || []);
    // Populate the Only document selector
    if (onlyDocSelect) {
      const prev = onlyDocSelect.value;
      const docs = (data.documents || []).map((d) => String(d.name || '').trim()).filter(Boolean);
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

async function removeLibraryDocument(name) {
  try {
    const resp = await fetch(`${API_BASE}/library/${encodeURIComponent(name)}`, {
      method: 'DELETE',
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
        recordConversationEntry('', summaryHtml, null);
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
    ingestLog.textContent = 'Select one or more PDFs.';
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
  const el = createMessageElement('user', text, { animate: options.animate !== false });
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

function maybeInsertSummaryMessage(summaryHtml) {
  const hasAny = chatHistory.length > 0 || chatWindow.innerHTML.trim().length > 0;
  if (!hasAny && summaryHtml) {
    appendBotMessage(summaryHtml);
  }
}

let askPollHandle = null;
async function askQuestion(payload, question) {
  if (askLog) askLog.textContent = 'Working…';
  const sendBtn = askForm?.querySelector('.send-btn');
  if (sendBtn) sendBtn.disabled = true;
  if (askProgress) askProgress.classList.add('active');
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
          if (sendBtn) sendBtn.disabled = false;
          return;
        }
        const d = await direct.json();
        if (askLog) { askLog.textContent = d.log || 'Ready.'; try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
        if (d.answer) {
          appendBotMessage(d.answer);
        }
        if (question) {
          recordConversationEntry(question, d.answer || '', d.answer_markdown);
        }
        if (d.answer) {
          const plain = stripHTML(d.answer);
          maybeUpdateTitleFrom(question + ' \n' + plain);
        }
      } catch (e) {
        if (askLog) askLog.textContent = `Error: ${e}`;
      }
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
      return;
    } else {
      if (askLog) askLog.textContent = `Error: ${text}`;
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
  const poll = async () => {
    try {
      const s = await fetch(`${API_BASE}/ask/status/${jobId}`);
      if (!s.ok) {
        if (s.status === 404) {
          // Fallback: server missing status route; perform direct ask once.
          clearInterval(askPollHandle); askPollHandle = null;
          try {
            const direct = await fetch(`${API_BASE}/ask`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload),
            });
            if (!direct.ok) {
              const t = await direct.text();
              if (askLog) askLog.textContent = `Error: ${t}`;
            } else {
              const d = await direct.json();
              if (askLog) { askLog.textContent = d.log || 'Ready.'; try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
              if (d.answer) {
                appendBotMessage(d.answer);
              }
              if (question) {
                recordConversationEntry(question, d.answer || '', d.answer_markdown);
              }
              if (d.answer) {
                const plain = stripHTML(d.answer);
                maybeUpdateTitleFrom(question + ' \n' + plain);
              }
            }
          } catch (e) {
            if (askLog) askLog.textContent = `Error: ${e}`;
          }
          if (sendBtn) sendBtn.disabled = false;
          if (askProgress) askProgress.classList.remove('active');
          return;
        } else {
          const txt = await s.text();
          if (askLog) askLog.textContent = `Error: ${txt}`;
          clearInterval(askPollHandle); askPollHandle = null;
          if (sendBtn) sendBtn.disabled = false;
          if (askProgress) askProgress.classList.remove('active');
          return;
        }
      }
      const st = await s.json();
      if (Array.isArray(st.logs) && askLog) { askLog.textContent = st.logs.join('\n'); try { askLog.scrollTop = askLog.scrollHeight; } catch {} }
      if (st.status === 'done') {
        clearInterval(askPollHandle); askPollHandle = null;
        if (st.answer) {
          appendBotMessage(st.answer);
        }
        if (question) {
          recordConversationEntry(question, st.answer || '', st.answer_markdown);
        }
        if (st.answer) {
          const plain = stripHTML(st.answer);
          maybeUpdateTitleFrom(question + ' \n' + plain);
        }
        if (sendBtn) sendBtn.disabled = false;
        if (askProgress) askProgress.classList.remove('active');
      } else if (st.status === 'error' || st.status === 'cancelled') {
        clearInterval(askPollHandle); askPollHandle = null;
        if (askLog) askLog.textContent += `\n${st.error || st.status}`;
        if (sendBtn) sendBtn.disabled = false;
        if (askProgress) askProgress.classList.remove('active');
      }
    } catch (err) {
      if (askLog) askLog.textContent = `Error: ${err}`;
      clearInterval(askPollHandle); askPollHandle = null;
      if (sendBtn) sendBtn.disabled = false;
      if (askProgress) askProgress.classList.remove('active');
    }
  };
  poll();
  askPollHandle = setInterval(poll, 800);
}

askForm?.addEventListener('submit', (evt) => {
  evt.preventDefault();
  const question = document.getElementById('question').value.trim();
  if (!question) {
    if (askLog) askLog.textContent = 'Ask a question first.';
    return;
  }
  // Immediately show user's message in the chat window
  appendUserMessage(question);
  const payload = {
    question,
    memory_enabled: true,
    formula_mode: document.getElementById('formula-mode').checked,
    agents_enabled: document.getElementById('agents-enabled').checked,
    web_enabled: document.getElementById('web-enabled').checked,
    exhaustive: document.getElementById('exhaustive').checked,
    top_k: Number(document.getElementById('top-k').value || 10),
    max_batches: Number(document.getElementById('max-batches').value || 6),
    time_budget: Number(document.getElementById('time-budget').value || 120),
    reranker: document.getElementById('reranker').value || 'off',
    history: chatHistory,
  };
  if (onlyDocSelect && onlyDocSelect.value) {
    payload.only_doc = onlyDocSelect.value;
  }
  askQuestion(payload, question);
  const qEl = document.getElementById('question');
  if (qEl) qEl.value = '';
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
  const payload = {
    openai_key: document.getElementById('openai-key').value,
    hf_token: document.getElementById('hf-token').value,
    serp_key: document.getElementById('serp-key').value,
    brave_key: document.getElementById('brave-key').value,
    openai_model: document.getElementById('openai-model').value,
    hf_model: document.getElementById('hf-model').value,
    embed_backend: document.getElementById('embed-backend').value,
    llm_backend: document.getElementById('llm-backend').value,
    memory_enabled: document.getElementById('settings-memory').checked,
    memory_tokens: Number(document.getElementById('memory-tokens').value || 1200),
    memory_file_mb: Number(document.getElementById('memory-file-mb').value || 50),
    openai_tpm: Number(document.getElementById('openai-tpm').value || 0),
    openai_rpm: Number(document.getElementById('openai-rpm').value || 0),
    ask_char_budget: Number(document.getElementById('ask-char-budget').value || 12000),
    ask_max_batches: Number(document.getElementById('ask-max-batches').value || 6),
    ask_time_budget: Number(document.getElementById('ask-time-budget').value || 120),
    ask_exhaustive: document.getElementById('settings-exhaustive').checked,
    ask_reranker: document.getElementById('settings-reranker').value || 'off',
    ask_candidates: Number(document.getElementById('ask-candidates').value || 300),
  };
  const resp = await fetch(`${API_BASE}/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!resp.ok) {
    const text = await resp.text();
    settingsStatus.textContent = `Error: ${text}`;
    return;
  }
  const data = await resp.json();
  settingsStatus.textContent = data.message || 'Saved.';
  if (data.defaults) {
    applyDefaults(data.defaults);
  }
  await refreshSettingsStatus();
});

async function refreshSettingsStatus() {
  try {
    const resp = await fetch(`${API_BASE}/settings`);
    if (!resp.ok) return;
    const data = await resp.json();
    if (data.defaults) {
      applyDefaults(data.defaults);
    }
    const keyStatusEl = document.getElementById('settings-status');
    if (keyStatusEl && data.keys) {
      keyStatusEl.textContent = [
        `OpenAI: ${data.keys.openai ? '✅' : '⚠️'}`,
        `HF: ${data.keys.hf ? '✅' : '⚠️'}`,
        `SerpAPI: ${data.keys.serpapi ? '✅' : '—'}`,
        `Brave: ${data.keys.brave ? '✅' : '—'}`,
      ].join('  ');
    }
  } catch (err) {
    console.warn('Failed to refresh settings', err);
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', refreshSettingsStatus);
} else {
  refreshSettingsStatus();
}

// -------------------------------------------------
// Conversations (localStorage based)
function escapeHTML(s) {
  return String(s || '').replace(/[&<>]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[ch]));
}

function saveConversations() {
  try {
    const payload = { active: activeConvId, conversations };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  } catch {}
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
    const resp = await fetch('/api/chat/title', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: String(text || '').slice(0, 4000) }),
    });
    if (!resp.ok) return;
    const data = await resp.json();
    const title = (data.title || '').trim();
    if (title) renameConversation(id, title);
  } catch {}
}

function recordConversationEntry(question, answerHtml, answerMarkdown) {
  if (!answerHtml && !answerMarkdown) return;
  const htmlValue = (answerHtml && answerHtml.trim())
    ? answerHtml
    : (answerMarkdown ? renderMarkdownToHtml(answerMarkdown) : '');
  const markdownValue = (answerMarkdown && answerMarkdown.trim())
    ? answerMarkdown
    : (htmlValue ? stripHTML(htmlValue) : '');
  const entry = {
    q: question,
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
    const markdown = item.a_markdown || item.a || '';
    const html = item.a_html || (markdown ? renderMarkdownToHtml(markdown) : '');
    if (question) lastEl = appendUserMessage(question, { scroll: false, animate: false });
    if (html) lastEl = appendBotMessage(html, { scroll: false, typeset: false, animate: false });
    normalized.push({ q: question, a: markdown, a_markdown: markdown, a_html: html });
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
    activeConvId = Object.keys(conversations)[0] || null;
  }
  saveConversations();
  renderConversationList();
  loadConversation(activeConvId);
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
  try { updateScrollDownBtn(); } catch {}
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
    const resp = await fetch('/api/library/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ names }),
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
    const resp = await fetch('/api/library/clear', { method: 'POST' });
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
