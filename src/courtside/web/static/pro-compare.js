/* Compare page picker (stats_compare.html): live suggestions for the "Add a player / team"
   box from /api/search (the top-bar search index). Picking a suggestion submits the form with
   add=p:<id> or add=t:<CODE>; the server adds it and redirects to the clean URL. Without JS the
   form still works: the server resolves the typed text with the same search. */
(function () {
  "use strict";
  var form = document.querySelector(".cmp-add");
  if (!form) return;
  var input = form.querySelector("#cmp-q"), list = form.querySelector("#cmp-list");
  var mode = form.getAttribute("data-mode") || "";
  var have = (form.getAttribute("data-have") || "").split(/\s+/).filter(Boolean);
  var timer = null, seq = 0, items = [], active = -1;

  function close() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
    active = -1;
  }
  function pick(r) {
    input.value = (r.type === "team" ? "t:" : "p:") + r.id;
    form.submit();
  }
  function setActive(i) {
    var lis = list.querySelectorAll("[role=option]");
    if (!lis.length) return;
    active = (i + lis.length) % lis.length;
    lis.forEach(function (li, j) { li.setAttribute("aria-selected", j === active ? "true" : "false"); });
    input.setAttribute("aria-activedescendant", lis[active].id);
    lis[active].scrollIntoView({ block: "nearest" });
  }
  function render(results, q) {
    items = results.filter(function (r) {
      if (mode === "p" && r.type !== "player") return false;
      if (mode === "t" && r.type !== "team") return false;
      return have.indexOf(String(r.id)) < 0;
    });
    active = -1;
    list.textContent = "";
    if (!items.length) {
      var none = document.createElement("li");
      none.className = "none";
      none.textContent = "No " + (mode === "t" ? "teams" : mode === "p" ? "players" : "players or teams") + " match “" + q + "”";
      list.appendChild(none);
    }
    items.forEach(function (r, i) {
      var li = document.createElement("li");
      li.id = "cmp-opt-" + i;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", "false");
      [["k", r.type === "team" ? "Team" : "Player"], ["n", r.name], ["s", r.type === "team" ? r.id : r.sub]]
        .forEach(function (p) {
          var s = document.createElement("span");
          s.className = p[0];
          s.textContent = p[1] || "";
          li.appendChild(s);
        });
      li.addEventListener("mousedown", function (e) { e.preventDefault(); pick(r); });
      list.appendChild(li);
    });
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }
  function query() {
    var q = input.value.trim(), my = ++seq;
    if (!q || q.indexOf(":") > 0) { close(); return; }
    fetch("/api/search?q=" + encodeURIComponent(q), { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : { results: [] }; })
      .then(function (d) { if (my === seq && input.value.trim()) render(d.results || [], q); })
      .catch(function () { /* offline: plain form submit still works */ });
  }
  input.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(query, 150); });
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); if (list.hidden) query(); else setActive(active + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1); }
    else if (e.key === "Enter") {
      var r = items[active >= 0 ? active : 0];
      if (r && !list.hidden) { e.preventDefault(); pick(r); }
    } else if (e.key === "Escape") { close(); }
  });
  input.addEventListener("blur", function () { setTimeout(close, 120); });
})();
