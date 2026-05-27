// =============================================================================
// tutorial.js — shared interactivity for every chapter page.
//   - Reading-progress bar
//   - Sidebar "current section" highlighting via IntersectionObserver
//   - Copy-to-clipboard buttons on every <pre>
//   - Mermaid init (deferred so dark theme matches)
// =============================================================================
(function () {
  // ---- 1. Progress bar -----------------------------------------------------
  const bar = document.createElement('div');
  bar.id = 'progress';
  document.body.appendChild(bar);
  const setProgress = () => {
    const h = document.documentElement;
    const pct = (h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight)) * 100;
    bar.style.width = pct + '%';
  };
  document.addEventListener('scroll', setProgress, { passive: true });
  setProgress();

  // ---- 2. Highlight current chapter in sidebar ----------------------------
  const sidebarLinks = document.querySelectorAll('#sidebar nav a');
  const here = location.pathname.split('/').filter(Boolean).pop() || 'index.html';
  sidebarLinks.forEach(a => {
    const href = a.getAttribute('href') || '';
    if (href.includes(here) || (here === '' && href.endsWith('index.html'))) {
      a.classList.add('current');
    }
  });

  // ---- 3. Copy-to-clipboard buttons ---------------------------------------
  // Use Clipboard API where available, fall back to document.execCommand for
  // older Safari on file:// or other restricted contexts.
  const copyText = (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise((resolve, reject) => {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      try { document.execCommand('copy'); resolve(); }
      catch (e) { reject(e); }
      finally { document.body.removeChild(ta); }
    });
  };

  document.querySelectorAll('pre').forEach(pre => {
    if (pre.classList.contains('mermaid')) return;
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'copy';
    btn.addEventListener('click', () => {
      const code = pre.querySelector('code') || pre;
      copyText(code.innerText).then(() => {
        btn.textContent = 'copied ✓';
        btn.classList.add('ok');
        setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('ok'); }, 1500);
      }).catch(() => {
        btn.textContent = 'press Ctrl+C';
      });
    });
    pre.appendChild(btn);
  });

  // ---- 4. Mermaid init with matching light theme --------------------------
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: true,
      theme: 'default',
      themeVariables: {
        background: '#ffffff',
        primaryColor: '#f6f8fa',
        primaryTextColor: '#1f2328',
        primaryBorderColor: '#0969da',
        lineColor: '#6b7280',
        secondaryColor: '#ddf4ff',
        tertiaryColor: '#f6f8fa',
        fontFamily: '-apple-system, "PingFang SC", "Segoe UI", sans-serif',
      },
      flowchart: { curve: 'basis' },
    });
  }
})();
