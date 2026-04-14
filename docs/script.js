/* ============================================================
   z-vector-search — GitHub Pages Interactions
   ============================================================ */

(function () {
  'use strict';

  /* ---------- Navbar scroll effect ---------- */
  const navbar = document.getElementById('navbar');
  let lastScroll = 0;

  function handleNavbarScroll() {
    const scrollY = window.scrollY;
    if (scrollY > 60) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    lastScroll = scrollY;
  }

  window.addEventListener('scroll', handleNavbarScroll, { passive: true });

  /* ---------- Mobile hamburger ---------- */
  const hamburger = document.getElementById('navHamburger');
  const navLinks = document.getElementById('navLinks');

  if (hamburger && navLinks) {
    hamburger.addEventListener('click', function () {
      hamburger.classList.toggle('open');
      navLinks.classList.toggle('open');
      document.body.style.overflow = navLinks.classList.contains('open') ? 'hidden' : '';
    });

    // Close menu on link click
    navLinks.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        hamburger.classList.remove('open');
        navLinks.classList.remove('open');
        document.body.style.overflow = '';
      });
    });
  }

  /* ---------- Smooth scroll for anchor links ---------- */
  document.querySelectorAll('a[href^="#"]').forEach(function (anchor) {
    anchor.addEventListener('click', function (e) {
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        e.preventDefault();
        const offset = navbar.offsetHeight + 20;
        const top = target.getBoundingClientRect().top + window.scrollY - offset;
        window.scrollTo({ top: top, behavior: 'smooth' });
      }
    });
  });

  /* ---------- Scroll reveal (IntersectionObserver) ---------- */
  const revealElements = document.querySelectorAll('.reveal');

  if ('IntersectionObserver' in window) {
    const revealObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            revealObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.12, rootMargin: '0px 0px -40px 0px' }
    );

    revealElements.forEach(function (el) {
      revealObserver.observe(el);
    });
  } else {
    // Fallback: show all
    revealElements.forEach(function (el) {
      el.classList.add('visible');
    });
  }

  /* ---------- Pipeline step animation ---------- */
  const pipelineSteps = document.querySelectorAll('.pipeline-step');

  if ('IntersectionObserver' in window && pipelineSteps.length > 0) {
    const pipelineObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            // Stagger reveal each step
            const stepIndex = parseInt(entry.target.getAttribute('data-step'), 10);
            setTimeout(function () {
              entry.target.classList.add('visible');
            }, stepIndex * 120);
            pipelineObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.2 }
    );

    pipelineSteps.forEach(function (step) {
      pipelineObserver.observe(step);
    });
  }

  /* ---------- Demo terminal line-by-line reveal ---------- */
  const demoBody = document.getElementById('demoBody');

  if (demoBody && 'IntersectionObserver' in window) {
    const demoLines = demoBody.querySelectorAll('.line');
    let demoRevealed = false;

    const demoObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting && !demoRevealed) {
            demoRevealed = true;
            demoLines.forEach(function (line, index) {
              setTimeout(function () {
                line.classList.add('visible');
              }, index * 100);
            });
            demoObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.25 }
    );

    demoObserver.observe(demoBody);
  }

  /* ---------- Active nav link highlighting ---------- */
  const sections = document.querySelectorAll('section[id]');
  const navAnchors = document.querySelectorAll('.nav-links a[href^="#"]');

  function updateActiveNav() {
    const scrollY = window.scrollY + navbar.offsetHeight + 100;
    let currentSection = '';

    sections.forEach(function (section) {
      if (section.offsetTop <= scrollY) {
        currentSection = section.getAttribute('id');
      }
    });

    navAnchors.forEach(function (anchor) {
      anchor.style.color = '';
      if (anchor.getAttribute('href') === '#' + currentSection) {
        anchor.style.color = 'var(--accent-cyan)';
      }
    });
  }

  window.addEventListener('scroll', updateActiveNav, { passive: true });

  /* ---------- Stats counter animation ---------- */
  const statValues = document.querySelectorAll('.stat-value');

  if ('IntersectionObserver' in window && statValues.length > 0) {
    const statsObserver = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            animateStatValue(entry.target);
            statsObserver.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.5 }
    );

    statValues.forEach(function (stat) {
      statsObserver.observe(stat);
    });
  }

  function animateStatValue(el) {
    const text = el.textContent.trim();

    // Only animate pure numbers
    var match = text.replace(/,/g, '').match(/^(\d+)$/);
    if (!match) return;

    var target = parseInt(match[1], 10);
    var duration = 1500;
    var start = performance.now();

    function update(now) {
      var progress = Math.min((now - start) / duration, 1);
      // Ease out cubic
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = Math.floor(target * eased);
      el.textContent = current.toLocaleString();
      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        el.textContent = target.toLocaleString();
      }
    }

    requestAnimationFrame(update);
  }

})();
