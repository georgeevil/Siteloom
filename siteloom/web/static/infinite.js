/* Continuous scroll for the console's lists (CLD-104).
 *
 * Progressive enhancement over a link that already works. The server
 * renders a real "Load more" anchor whose href carries the cursor, so a
 * client with no JavaScript — or one where this file failed to load —
 * still walks the whole list one click at a time. This only removes the
 * clicking.
 *
 * Three things here are deliberate:
 *
 *  - The observer is given the list's own scrolling ancestor as `root`.
 *    The events list scrolls inside its own container, not the page, and
 *    an observer left on the default root watches the viewport: the
 *    sentinel is either always intersecting (fires forever) or never is
 *    (fires not at all), depending on layout. Neither failure looks like
 *    a bug in scrolling.
 *  - A failed fetch restores the button and says so. Silently swallowing
 *    it leaves a list that has simply stopped, which reads as "that is
 *    everything" — the one thing this must never say untruthfully.
 *  - Only one request is in flight at a time. The sentinel crosses the
 *    threshold on every scroll event during a slow fetch, and without the
 *    guard each crossing appends the same page again.
 */
(function () {
  "use strict";

  function scrollParent(node) {
    // The element that actually scrolls this list. Walking up beats
    // hard-coding a selector: three screens use this and their markup is
    // not the same shape.
    for (var el = node.parentElement; el; el = el.parentElement) {
      var overflow = getComputedStyle(el).overflowY;
      if ((overflow === "auto" || overflow === "scroll") &&
          el.scrollHeight > el.clientHeight) {
        return el;
      }
    }
    return null; // the page itself; IntersectionObserver's default root
  }

  function setup(root) {
    var button = root.querySelector("[data-load-more]");
    if (!button) return;

    var list = document.querySelector(root.dataset.infiniteTarget);
    if (!list) return;

    var busy = false;

    function fail(message) {
      busy = false;
      root.dataset.state = "error";
      button.hidden = false;
      button.textContent = button.dataset.retryLabel || "Retry";
      var note = root.querySelector("[data-load-error]");
      if (note) {
        note.textContent = message;
        note.hidden = false;
      }
    }

    function load() {
      if (busy || !button.href) return;
      busy = true;
      root.dataset.state = "loading";
      button.hidden = true;
      var note = root.querySelector("[data-load-error]");
      if (note) note.hidden = true;

      fetch(button.href, {headers: {"X-Requested-With": "fetch"}})
        .then(function (response) {
          if (!response.ok) throw new Error("HTTP " + response.status);
          return response.text();
        })
        .then(function (html) {
          var parsed = new DOMParser().parseFromString(html, "text/html");
          var rows = parsed.querySelector(root.dataset.infiniteTarget);
          if (rows) list.append.apply(list, Array.from(rows.children));

          // The next cursor arrives with the rows it continues, so the
          // fragment carries its own footer rather than the client
          // guessing what comes after what it just appended.
          var next = parsed.querySelector("[data-load-more]");
          if (next && next.getAttribute("href")) {
            button.href = next.getAttribute("href");
            button.hidden = false;
            button.textContent = next.textContent;
            root.dataset.state = "idle";
          } else {
            button.remove();
            root.dataset.state = "end";
            var done = root.querySelector("[data-load-end]");
            if (done) done.hidden = false;
          }
          busy = false;
        })
        .catch(function (error) {
          fail("Could not load more — " + error.message + ".");
        });
    }

    button.addEventListener("click", function (event) {
      event.preventDefault();
      load();
    });

    var sentinel = root.querySelector("[data-load-sentinel]");
    if (!sentinel || !("IntersectionObserver" in window)) return;

    new IntersectionObserver(function (entries) {
      if (entries.some(function (e) { return e.isIntersecting; })) load();
    }, {
      root: scrollParent(sentinel),
      // Start fetching before the operator reaches the end, so a fast
      // scroll does not stall on an empty list.
      rootMargin: "400px 0px",
    }).observe(sentinel);
  }

  function init() {
    document.querySelectorAll("[data-infinite]").forEach(setup);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
