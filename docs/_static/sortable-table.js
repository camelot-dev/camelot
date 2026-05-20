/* Tiny dependency-free sortable-table helper.
 *
 * Attaches click-to-sort to any <table class="sortable">. First click on
 * a header sorts ascending, second click descending, third resets.
 * Numeric, date, and string cells are detected automatically.
 *
 * Adapted from the public-domain pattern at https://adrianroselli.com/
 * — kept minimal (~80 lines, no deps) so it ships alongside the docs
 * without an npm install.
 */
(function () {
  "use strict";

  function parseCell(text) {
    // Strip surrounding whitespace.
    text = (text || "").trim();
    if (!text) return { kind: "empty", value: "" };
    // ISO date.
    const isoDate = /^\d{4}-\d{2}-\d{2}$/.test(text);
    if (isoDate) return { kind: "date", value: text };
    // Number (allow %, comma separators, leading minus).
    const num = text.replace(/[\s,%]/g, "");
    if (/^-?\d+(\.\d+)?$/.test(num)) {
      return { kind: "number", value: parseFloat(num) };
    }
    return { kind: "string", value: text.toLowerCase() };
  }

  function compareCells(a, b) {
    // Empty cells sort to the bottom regardless of direction.
    if (a.kind === "empty" && b.kind === "empty") return 0;
    if (a.kind === "empty") return 1;
    if (b.kind === "empty") return -1;
    if (a.kind === "number" && b.kind === "number") return a.value - b.value;
    if (a.kind === "date" && b.kind === "date") {
      return a.value < b.value ? -1 : a.value > b.value ? 1 : 0;
    }
    // Mixed or string: lexicographic.
    return a.value < b.value ? -1 : a.value > b.value ? 1 : 0;
  }

  function makeSortable(table) {
    const headers = table.querySelectorAll("thead th");
    if (!headers.length) return;
    const tbody = table.querySelector("tbody");
    if (!tbody) return;

    // Cache the original row order so a third click can restore it.
    const originalRows = Array.from(tbody.rows);

    headers.forEach(function (th, colIdx) {
      th.addEventListener("click", function () {
        const current = th.getAttribute("aria-sort");
        let next;
        if (!current || current === "none") next = "ascending";
        else if (current === "ascending") next = "descending";
        else next = "none";

        // Clear all headers, set the active one.
        headers.forEach(function (other) {
          other.setAttribute("aria-sort", "none");
        });
        th.setAttribute("aria-sort", next);

        if (next === "none") {
          // Restore original.
          originalRows.forEach(function (r) {
            tbody.appendChild(r);
          });
          return;
        }

        const rows = Array.from(tbody.rows);
        rows.sort(function (a, b) {
          const av = parseCell(a.cells[colIdx]?.textContent);
          const bv = parseCell(b.cells[colIdx]?.textContent);
          const cmp = compareCells(av, bv);
          return next === "ascending" ? cmp : -cmp;
        });
        rows.forEach(function (r) {
          tbody.appendChild(r);
        });
      });
    });
  }

  function init() {
    document.querySelectorAll("table.sortable").forEach(makeSortable);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
