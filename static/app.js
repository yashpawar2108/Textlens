// app.js — TextLens frontend logic
(() => {
  const API_BASE = '/api/v1/hackrx';
  const realFile = document.getElementById('realFile');
  const uploadBtn = document.getElementById('uploadBtn');
  const openChat = document.getElementById('openChat');
  const messages = document.getElementById('messages');
  const msgInput = document.getElementById('msgInput');
  const sendBtn = document.getElementById('sendBtn');
  const clearDoc = document.getElementById('clearDoc');
  const viewDoc = document.getElementById('viewDoc');
  const clearCache = document.getElementById('clearCache');
  const uploadedFileName = document.getElementById('uploadedFileName');
  const saveToken = document.getElementById('saveToken');
  const tokenField = document.getElementById('tokenField');
  const themeToggle = document.getElementById('themeToggle');
  const copyBtn = document.getElementById('copyBtn');
  const emptyState = document.getElementById('emptyState');
  const statusIndicator = document.getElementById('statusIndicator');

  // init
  document.getElementById('year').textContent = new Date().getFullYear();
  tokenField.value = localStorage.getItem('HACKRX_TOKEN') || '';
  let lastBotText = '';

  function setStatus(text){ statusIndicator.textContent = text; }
  setStatus('Ready');

  // theme
  function applyTheme(theme){ document.body.className = theme === 'dark' ? 'theme-dark' : 'theme-light'; }
  const savedTheme = localStorage.getItem('TEXTLENS_THEME') || 'light'; applyTheme(savedTheme);
  themeToggle.addEventListener('click', ()=>{ const t = document.body.classList.contains('theme-dark') ? 'light' : 'dark'; applyTheme(t); localStorage.setItem('TEXTLENS_THEME', t); });

  // token
  saveToken.onclick = () => { const t = tokenField.value.trim(); if(!t) return alert('Paste token'); localStorage.setItem('HACKRX_TOKEN', t); alert('Token saved locally'); };
  function getAuthHeader(){ const t = localStorage.getItem('HACKRX_TOKEN'); if(!t) throw new Error('Missing bearer token. Save it in the header area.'); return { 'Authorization': 'Bearer ' + t }; }

  // upload
  uploadBtn.addEventListener('click', ()=> realFile.click());
  realFile.addEventListener('change', async (e) => { const f = e.target.files[0]; if(f) await uploadFile(f); });

  async function uploadFile(file){
    if(!file.name.toLowerCase().endsWith('.pdf')) return alert('Only PDFs allowed');
    uploadedFileName.textContent = `Uploading: ${file.name}`;
    const fd = new FormData(); fd.append('file', file);
    try{
      const res = await fetch(API_BASE + '/upload', { method:'POST', body: fd, headers: getAuthHeader() });
      if(!res.ok){ const err = await res.json().catch(()=>null); throw new Error(err?.detail || res.statusText); }
      const data = await res.json(); uploadedFileName.innerHTML = `Uploaded: <b>${data.filename}</b>`; addSystemMessage(`Uploaded ${data.filename}`);
    }catch(err){ uploadedFileName.textContent = 'Upload failed'; addSystemMessage(`Upload failed: ${err.message}`); }
  }

  // clear/view/cache
  clearDoc.addEventListener('click', ()=>{ uploadedFileName.textContent = 'No document uploaded'; addSystemMessage('Document cleared locally'); });
  viewDoc.addEventListener('click', ()=>{ alert('To view the raw uploaded file, open server cache dir or re-upload.'); });
  clearCache.addEventListener('click', async ()=>{
    try{ const res = await fetch(API_BASE + '/cache/clear', { method:'DELETE', headers: getAuthHeader() }); if(!res.ok) throw new Error('Unable to clear cache'); addSystemMessage('Server cache cleared'); }catch(e){ addSystemMessage('Clear cache failed: ' + e.message); }
  });

  // messaging
  function addSystemMessage(text){ emptyState.style.display='none'; const el = document.createElement('div'); el.className='msg bot'; el.innerHTML = `<div class="avatar">TL</div><div class="bubble"><em>${text}</em></div>`; messages.appendChild(el); scrollToBottom(); }
  function addUserMessage(text){ emptyState.style.display='none'; const el = document.createElement('div'); el.className='msg user'; el.innerHTML = `<div class="avatar">YOU</div><div class="bubble">${escapeHtml(text)}</div>`; messages.appendChild(el); scrollToBottom(); }
  function addBotMessage(text){ lastBotText = text; emptyState.style.display='none'; const el = document.createElement('div'); el.className='msg bot'; el.innerHTML = `<div class="avatar">TL</div><div class="bubble">${formatResponse(text)}</div>`; messages.appendChild(el); scrollToBottom(); }

  function formatResponse(text){ // basic formatting and small source highlighting
    const safe = escapeHtml(text).replace(/\n/g, '<br>');
    return safe;
  }

  function escapeHtml(unsafe){ return unsafe.replace(/[&<"'`=\/]/g, function(s){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;','`':'&#96;','=':'&#61;'})[s]; }); }

  function scrollToBottom(){ const w = document.getElementById('chatWindow'); w.scrollTop = w.scrollHeight; }

  // send
  sendBtn.addEventListener('click', sendMessage);
  msgInput.addEventListener('keydown', (e)=>{ if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); sendMessage(); } });

  async function sendMessage(){ const text = msgInput.value.trim(); if(!text) return; msgInput.value=''; addUserMessage(text); try{
    setStatus('Thinking...');
    const res = await fetch(API_BASE + '/chat', { method:'POST', headers: { 'Content-Type': 'application/json', ...getAuthHeader() }, body: JSON.stringify({ message: text }) });
    if(!res.ok){ const err = await res.json().catch(()=>null); throw new Error(err?.detail || res.statusText); }
    const data = await res.json(); addBotMessage(data.response || '(no response)'); setStatus('Ready');
  }catch(e){ addSystemMessage('Request failed: ' + e.message); setStatus('Error'); }
  }

  copyBtn.addEventListener('click', ()=>{ if(!lastBotText) return alert('No response to copy'); navigator.clipboard.writeText(lastBotText).then(()=> alert('Copied'), ()=> alert('Copy failed')); });

  // helper: initial health check
  (async function health(){ try{ const r = await fetch('/health'); if(r.ok) setStatus('Connected'); else setStatus('Unhealthy'); }catch(e){ setStatus('Offline'); addSystemMessage('Backend unreachable — start server'); } })();

})();
