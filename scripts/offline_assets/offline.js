// Replaces the interactive parts of the MyST theme for the file:// copy
// produced by scripts/make_offline.py. Plain script, no modules or fetch().
(function () {
  "use strict";
  var ROOT = window.OFFLINE_ROOT || "";

  // Sidebar folders.
  document.querySelectorAll('button[aria-label="Open Folder"]').forEach(function (btn) {
    btn.addEventListener("click", function () {
      var open = btn.getAttribute("aria-expanded") !== "true";
      var state = open ? "open" : "closed";
      var content = document.getElementById(btn.getAttribute("aria-controls"));
      btn.setAttribute("aria-expanded", String(open));
      btn.setAttribute("data-state", state);
      btn.closest("div[data-state].w-full").setAttribute("data-state", state);
      if (content) {
        content.setAttribute("data-state", state);
        content.hidden = !open;
      }
    });
  });

  // Light / dark theme.
  document.querySelectorAll(".myst-theme-button").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var cls = document.documentElement.classList;
      var next = cls.contains("dark") ? "light" : "dark";
      cls.remove("light", "dark");
      cls.add(next);
      try { localStorage.setItem("myst:theme", next); } catch (e) {}
    });
  });

  // Sidebar on narrow screens.
  document.querySelectorAll(".myst-top-nav-menu-button").forEach(function (btn) {
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      document.body.classList.toggle("offline-sidebar-open");
    });
  });
  document.addEventListener("click", function (ev) {
    if (!document.body.classList.contains("offline-sidebar-open")) return;
    if (ev.target.closest(".myst-primary-sidebar")) return;
    document.body.classList.remove("offline-sidebar-open");
  });

  // Copy buttons on code blocks.
  document.querySelectorAll('button[aria-label="Copy code to clipboard"]').forEach(function (btn) {
    btn.addEventListener("click", function () {
      var code = btn.parentElement.querySelector("pre");
      if (!code) return;
      var text = code.innerText;
      var done = function () {
        btn.setAttribute("aria-pressed", "true");
        setTimeout(function () { btn.setAttribute("aria-pressed", "false"); }, 1500);
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done);
      } else {
        var ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        ta.remove();
        done();
      }
    });
  });

  // Search.
  var modal, input, list, loading = false;

  function loadIndex(cb) {
    if (window.OFFLINE_SEARCH) return cb();
    if (loading) return;
    loading = true;
    var s = document.createElement("script");
    s.src = ROOT + "_offline/search-index.js";
    s.onload = cb;
    document.head.appendChild(s);
  }

  function esc(t) {
    return t.replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  function headingOf(r) {
    var h = r.hierarchy || {}, parts = [];
    for (var i = 1; i <= 6; i++) if (h["lvl" + i]) parts.push(h["lvl" + i]);
    return parts;
  }

  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var hits = [];
    window.OFFLINE_SEARCH.forEach(function (r) {
      var head = headingOf(r).join(" ").toLowerCase();
      var body = (r.content || "").toLowerCase();
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var inHead = head.indexOf(terms[i]) >= 0, inBody = body.indexOf(terms[i]) >= 0;
        if (!inHead && !inBody) return;
        score += (inHead ? 10 : 0) + (inBody ? 1 : 0);
      }
      if (r.type && r.type !== "content") score += 2;
      hits.push({ r: r, score: score });
    });
    hits.sort(function (a, b) { return b.score - a.score; });
    // A heading and its section text share a URL; show them as one result.
    var seen = {}, out = [];
    hits.forEach(function (h) {
      var prev = seen[h.r.url];
      if (!prev) { seen[h.r.url] = h.r; out.push(h.r); }
      else if (!prev.content && h.r.content) {
        out[out.indexOf(prev)] = seen[h.r.url] = h.r;
      }
    });
    return out.slice(0, 40);
  }

  function snippet(r, q) {
    var text = r.content || "";
    if (!text) return "";
    var t = q.toLowerCase().split(/\s+/).filter(Boolean)[0] || "";
    var i = Math.max(0, text.toLowerCase().indexOf(t) - 60);
    var s = (i > 0 ? "…" : "") + text.slice(i, i + 180).replace(/\s+/g, " ");
    return esc(s);
  }

  function render() {
    var q = input.value.trim();
    if (!window.OFFLINE_SEARCH) {
      list.innerHTML = '<li class="offline-search-empty">Loading…</li>';
      return;
    }
    var res = search(q);
    if (!q) { list.innerHTML = ""; return; }
    if (!res.length) {
      list.innerHTML = '<li class="offline-search-empty">No results</li>';
      return;
    }
    list.innerHTML = res.map(function (r) {
      var parts = headingOf(r);
      return '<li><a href="' + esc(ROOT + r.url) + '"><div class="offline-search-title">' +
        esc(parts[parts.length - 1] || "") + '</div><div class="offline-search-path">' +
        esc(parts.slice(0, -1).join(" › ")) + '</div><div class="offline-search-snippet">' +
        snippet(r, q) + "</div></a></li>";
    }).join("");
  }

  function openSearch() {
    if (!modal) {
      modal = document.createElement("div");
      modal.className = "offline-search";
      modal.innerHTML = '<div class="offline-search-box"><input type="search" ' +
        'placeholder="Search the book" aria-label="Search"><ul></ul></div>';
      document.body.appendChild(modal);
      input = modal.querySelector("input");
      list = modal.querySelector("ul");
      input.addEventListener("input", render);
      modal.addEventListener("click", function (ev) { if (ev.target === modal) closeSearch(); });
      input.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter") {
          var a = list.querySelector("a");
          if (a) window.location.href = a.href;
        }
      });
    }
    modal.style.display = "flex";
    input.focus();
    input.select();
    loadIndex(render);
  }

  function closeSearch() { if (modal) modal.style.display = "none"; }

  document.querySelectorAll(".myst-search-bar").forEach(function (btn) {
    btn.addEventListener("click", openSearch);
  });
  document.addEventListener("keydown", function (ev) {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") {
      ev.preventDefault();
      openSearch();
    } else if (ev.key === "Escape") {
      closeSearch();
    }
  });
})();
