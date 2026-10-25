/* Courtside stats dashboard behaviour (no dependencies, no build step). Loaded on every page.

   1. Sortable tables: <table data-sortable>
      - Click (or Enter/Space on) a <th> to sort its column; a second click flips direction.
      - Sort key: the cell's data-v attribute when present (data-v="" = missing), else its text
        ($ , % + and the typographic minus are ignored, so "$11,400" and "−3.2" sort as numbers).
      - A column is numeric when every non-empty key parses as a number; numbers sort high-first
        on the first click, text A-Z. <th data-sort-first="asc|desc"> overrides.
      - Missing values always sink to the bottom. <th data-nosort> is not clickable.
      - <th data-sorted="desc|asc"> marks the column the server already sorted by (shows the arrow).
      - <table data-renumber>: cells with class "rownum" are renumbered 1..n after each sort.
      - Only tbody rows move (each tbody on its own); keep totals in <tfoot>. No colspans in
        the header row or body.
   2. Segmented controls that toggle panels:
        <span class="seg" data-seg-group="box">
          <button type="button" data-seg-target="trad" class="on">Traditional</button>
          <button type="button" data-seg-target="adv">Advanced</button></span>
        <div data-seg-panel="box" data-seg-value="trad">...</div>
        <div data-seg-panel="box" data-seg-value="adv" hidden>...</div>
      - data-seg-value may list several values ("trad dk"). Several controls can share a group.
      - Render the inactive panels with `hidden` so the page is right before JS runs.
      - <span ... data-seg-remember> keeps the choice per device (localStorage).
      - A "seg:change" event (detail: {group, value}) fires on document after each switch.
      - A seg made of <a href> links (server-side switch, e.g. ?season=) needs no JS.
   3. Tooltips: any element with data-tip="text" (newlines allowed) shows a hover/focus tip.
   4. Global search box in the top bar (#gsearch-q): /api/search?q= with a 150 ms debounce,
      arrow keys + Enter, Escape closes, "/" focuses it from anywhere on the page.
   window.CourtsideStats exposes {sortTable, initSortable, initSeg, rankColor, ordinal} for
   pages that build tables in JS. */
(function () {
  "use strict";
  var doc = document;

  // ------------------------------------------------------------------ helpers
  function ordinal(n) {
    var v = n % 100, s = ["th", "st", "nd", "rd"];
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }
  function mix(a, b, t) {
    var out = "#";
    for (var i = 1; i < 7; i += 2) {
      var x = parseInt(a.slice(i, i + 2), 16), y = parseInt(b.slice(i, i + 2), 16);
      out += ("0" + Math.round(x + (y - x) * t).toString(16)).slice(-2);
    }
    return out;
  }
  /* same palette as fmt.rank_color: 1st blue -> gray -> last red */
  function rankColor(rank, n) {
    n = Math.max(n || 30, 2);
    var t = Math.min(Math.max((rank - 1) / (n - 1), 0), 1);
    return t < 0.5 ? mix("#2a6fc4", "#3a3f4b", t * 2) : mix("#3a3f4b", "#c44a4a", (t - 0.5) * 2);
  }

  // ------------------------------------------------------------------ 1. sortable tables
  var NUM_RE = /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i;
  function cellKey(td) {
    if (!td) return "";
    var raw = td.hasAttribute("data-v") ? td.getAttribute("data-v") : td.textContent;
    raw = (raw || "").trim();
    return raw === "–" || raw === "—" || raw === "-" ? "" : raw;
  }
  function asNum(s) {
    var c = s.replace(/[−–]/g, "-").replace(/[$,%+\s]/g, "");
    return NUM_RE.test(c) ? parseFloat(c) : null;
  }
  function sortTable(table, col, dir) {
    var ths = headerCells(table);
    Array.prototype.forEach.call(table.tBodies, function (tb) {
      var rows = Array.prototype.slice.call(tb.rows);
      var keys = rows.map(function (r) { return cellKey(r.cells[col]); });
      var numeric = keys.every(function (k) { return k === "" || asNum(k) !== null; });
      var dec = rows.map(function (r, i) {
        var k = keys[i];
        return { r: r, i: i, empty: k === "", k: numeric ? asNum(k) : k };
      });
      dec.sort(function (a, b) {
        if (a.empty !== b.empty) return a.empty ? 1 : -1;
        if (a.empty) return a.i - b.i;
        var c = numeric ? a.k - b.k : a.k.localeCompare(b.k, undefined, { numeric: true, sensitivity: "base" });
        return c === 0 ? a.i - b.i : (dir === "asc" ? c : -c);
      });
      dec.forEach(function (d) { tb.appendChild(d.r); });
      if (table.hasAttribute("data-renumber")) {
        var n = 0;
        dec.forEach(function (d) {
          var cell = d.r.querySelector(".rownum");
          if (cell) cell.textContent = ++n;
        });
      }
    });
    ths.forEach(function (th, i) { mark(th, i === col ? dir : null); });
  }
  function headerCells(table) {
    var head = table.tHead;
    if (!head || !head.rows.length) return [];
    return Array.prototype.slice.call(head.rows[head.rows.length - 1].cells);
  }
  function mark(th, dir) {
    var ind = th.querySelector(".sort-ind");
    if (!dir) {
      th.removeAttribute("aria-sort");
      if (ind) ind.remove();
      return;
    }
    th.setAttribute("aria-sort", dir === "asc" ? "ascending" : "descending");
    if (!ind) {
      ind = doc.createElement("span");
      ind.className = "sort-ind";
      ind.setAttribute("aria-hidden", "true");
      th.appendChild(ind);
    }
    ind.textContent = dir === "asc" ? "▴" : "▾";
  }
  function initSortable(table) {
    if (table.__sortable) return;
    table.__sortable = true;
    var ths = headerCells(table);
    ths.forEach(function (th, col) {
      if (th.hasAttribute("data-sorted")) mark(th, th.getAttribute("data-sorted") === "asc" ? "asc" : "desc");
      if (th.hasAttribute("data-nosort")) return;
      th.tabIndex = 0;
      function go() {
        var cur = th.getAttribute("aria-sort");
        var dir;
        if (cur) dir = cur === "ascending" ? "desc" : "asc";
        else if (th.getAttribute("data-sort-first")) dir = th.getAttribute("data-sort-first");
        else {
          var tb = table.tBodies[0], numeric = true;
          if (tb) Array.prototype.forEach.call(tb.rows, function (r) {
            var k = cellKey(r.cells[col]);
            if (k !== "" && asNum(k) === null) numeric = false;
          });
          dir = numeric ? "desc" : "asc";
        }
        sortTable(table, col, dir);
      }
      th.addEventListener("click", go);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); }
      });
    });
  }
  doc.querySelectorAll("table[data-sortable]").forEach(initSortable);

  // ------------------------------------------------------------------ 2. segmented controls
  function selectSeg(group, value, remember) {
    doc.querySelectorAll('[data-seg-group="' + group + '"] [data-seg-target]').forEach(function (b) {
      var on = b.getAttribute("data-seg-target") === value;
      b.classList.toggle("on", on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
    doc.querySelectorAll('[data-seg-panel="' + group + '"]').forEach(function (p) {
      p.hidden = (p.getAttribute("data-seg-value") || "").split(/\s+/).indexOf(value) < 0;
    });
    if (remember) { try { localStorage.setItem("cs-seg-" + group, value); } catch (e) { /* private mode */ } }
    doc.dispatchEvent(new CustomEvent("seg:change", { detail: { group: group, value: value } }));
  }
  function initSeg(box) {
    if (box.__seg) return;
    box.__seg = true;
    var group = box.getAttribute("data-seg-group"), remember = box.hasAttribute("data-seg-remember");
    box.addEventListener("click", function (e) {
      var b = e.target.closest("[data-seg-target]");
      if (b && box.contains(b)) selectSeg(group, b.getAttribute("data-seg-target"), remember);
    });
    if (remember) {
      var saved = null;
      try { saved = localStorage.getItem("cs-seg-" + group); } catch (e) { /* private mode */ }
      if (saved && box.querySelector('[data-seg-target="' + saved + '"]')) selectSeg(group, saved, false);
    }
  }
  doc.querySelectorAll("[data-seg-group]").forEach(initSeg);

  // ------------------------------------------------------------------ 3. tooltips
  var tip = null, tipFor = null;
  function showTip(el, x, y) {
    if (!tip) {
      tip = doc.createElement("div");
      tip.className = "stat-tip";
      tip.setAttribute("role", "tooltip");
      doc.body.appendChild(tip);
    }
    tip.textContent = el.getAttribute("data-tip");
    tip.hidden = false;
    tipFor = el;
    moveTip(x, y);
  }
  function moveTip(x, y) {
    if (!tip || tip.hidden) return;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    var left = x + 14, top = y + 12;
    if (left + w > window.innerWidth - 8) left = Math.max(8, x - w - 10);
    if (top + h > window.innerHeight - 8) top = Math.max(8, y - h - 10);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hideTip() { if (tip) tip.hidden = true; tipFor = null; }
  doc.addEventListener("mouseover", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el && el !== tipFor) showTip(el, e.clientX, e.clientY);
    else if (!el && tipFor) hideTip();
  });
  doc.addEventListener("mousemove", function (e) { if (tipFor) moveTip(e.clientX, e.clientY); });
  doc.addEventListener("focusin", function (e) {
    var el = e.target.closest && e.target.closest("[data-tip]");
    if (el) { var r = el.getBoundingClientRect(); showTip(el, r.left, r.bottom); }
  });
  doc.addEventListener("focusout", hideTip);
  window.addEventListener("scroll", hideTip, { passive: true });

  // ------------------------------------------------------------------ 4. global search
  var input = doc.getElementById("gsearch-q"), list = doc.getElementById("gsearch-list");
  var bar = doc.querySelector(".topbar"), toggle = doc.querySelector(".search-toggle");
  if (input && list) {
    var timer = null, seq = 0, items = [], active = -1;
    var close = function () {
      list.hidden = true;
      input.setAttribute("aria-expanded", "false");
      input.removeAttribute("aria-activedescendant");
      active = -1;
    };
    var setActive = function (i) {
      var lis = list.querySelectorAll("[role=option]");
      if (!lis.length) return;
      active = (i + lis.length) % lis.length;
      lis.forEach(function (li, j) { li.setAttribute("aria-selected", j === active ? "true" : "false"); });
      input.setAttribute("aria-activedescendant", lis[active].id);
      lis[active].scrollIntoView({ block: "nearest" });
    };
    var render = function (results, q) {
      items = results;
      active = -1;
      list.textContent = "";
      if (!results.length) {
        var none = doc.createElement("li");
        none.className = "gs-none";
        none.textContent = "No players or teams match “" + q + "”";
        list.appendChild(none);
      }
      results.forEach(function (r, i) {
        var li = doc.createElement("li"), a = doc.createElement("a");
        li.id = "gs-opt-" + i;
        li.setAttribute("role", "option");
        li.setAttribute("aria-selected", "false");
        a.href = r.url;
        a.tabIndex = -1;
        [["gs-k", r.type === "team" ? "Team" : "Player"], ["gs-n", r.name], ["gs-s", r.type === "team" ? r.id : r.sub]]
          .forEach(function (p) {
            var s = doc.createElement("span");
            s.className = p[0];
            s.textContent = p[1] || "";
            a.appendChild(s);
          });
        li.appendChild(a);
        list.appendChild(li);
      });
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
    };
    var query = function () {
      var q = input.value.trim(), my = ++seq;
      if (!q) { items = []; close(); return; }
      fetch("/api/search?q=" + encodeURIComponent(q), { headers: { Accept: "application/json" } })
        .then(function (r) { return r.ok ? r.json() : { results: [] }; })
        .then(function (d) { if (my === seq && input.value.trim()) render(d.results || [], q); })
        .catch(function () { /* offline: leave the list as it is */ });
    };
    input.addEventListener("input", function () {
      clearTimeout(timer);
      timer = setTimeout(query, 150);
    });
    input.addEventListener("focus", function () { if (input.value.trim() && list.childNodes.length) { list.hidden = false; input.setAttribute("aria-expanded", "true"); } });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); if (list.hidden) query(); else setActive(active + 1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
      else if (e.key === "Enter") {
        var pick = items[active >= 0 ? active : 0];
        if (pick && !list.hidden) { e.preventDefault(); location.href = pick.url; }
      } else if (e.key === "Escape") {
        if (!list.hidden) close();
        else { input.blur(); if (bar) bar.classList.remove("is-searching"); }
      }
    });
    doc.addEventListener("click", function (e) {
      if (!e.target.closest(".gsearch") && !e.target.closest(".search-toggle")) close();
    });
    var openSearch = function () {
      if (bar) bar.classList.add("is-searching");
      if (toggle) toggle.setAttribute("aria-expanded", "true");
      input.focus();
      input.select();
    };
    if (toggle) toggle.addEventListener("click", function () {
      if (bar && bar.classList.contains("is-searching")) {
        bar.classList.remove("is-searching");
        toggle.setAttribute("aria-expanded", "false");
        close();
      } else openSearch();
    });
    doc.addEventListener("keydown", function (e) {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
      var t = e.target, tag = t && t.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || (t && t.isContentEditable)) return;
      e.preventDefault();
      openSearch();
    });
  }

  window.CourtsideStats = { sortTable: sortTable, initSortable: initSortable, initSeg: initSeg,
                            selectSeg: selectSeg, rankColor: rankColor, ordinal: ordinal };
})();
