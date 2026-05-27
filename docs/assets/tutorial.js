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

  // ---- 4. Mermaid init: paper / book theme (matches Hermes Tutorial) ------
  if (window.mermaid) {
    window.mermaid.initialize({
      startOnLoad: true,
      theme: 'neutral',
      themeVariables: {
        fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", "PingFang SC", sans-serif',
        fontSize: '14px',
        primaryColor:        '#fbfaf6',
        primaryTextColor:    '#2a2723',
        primaryBorderColor:  '#8b1538',
        lineColor:           '#4d4a44',
        secondaryColor:      '#f3f1e8',
        tertiaryColor:       '#e9e6d9',
        background:          '#fbfaf6',
        mainBkg:             '#fbfaf6',
        secondBkg:           '#f3f1e8',
        tertiaryTextColor:   '#2a2723',
        nodeBorder:          '#c2bea9',
        clusterBkg:          '#f6f4eb',
        clusterBorder:       '#d6d3c4',
        titleColor:          '#8b1538',
        edgeLabelBackground: '#fbfaf6',
        textColor:           '#2a2723',
        noteBkgColor:        '#f5ede0',
        noteBorderColor:     '#a86420',
        noteTextColor:       '#2a2723',
      },
      flowchart: { htmlLabels: true, curve: 'basis' },
    });
  }
})();
