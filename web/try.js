'use strict';
/* The Try it flow: four steps (your day, your energy, what to watch, your picks). Plain JavaScript, no framework. */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/* ---------------------------------------------------------- profiles (live on this device, with a secret token) */
const localProfiles = () => { try { return JSON.parse(localStorage.getItem('reelstate_profiles') || '[]'); } catch (_) { return []; } };
const saveProfiles = (p) => localStorage.setItem('reelstate_profiles', JSON.stringify(p));
const tokenOf = (uid) => (localProfiles().find(p => p.user_id === uid) || {}).token;

const state = {
  user: null, step: 1, furthest: 1, done: new Set(), events: [], lastKey: null, answers: [], quizStarted: false, quizShownAt: 0,
  runtime: '', language: '', genres: new Set(), group: new Set(), mood: null,
  activeRec: null, activeToken: null, activeWho: '', liked: null,
};

const api = async (path, body, token) => {
  const headers = {}; const tk = token ?? (state.user && state.user.token); if (tk) headers['X-Token'] = tk;
  const opts = body === undefined ? { headers } : { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json' }, body: JSON.stringify(body) };
  const r = await fetch('/api' + path, opts);
  if (!r.ok) throw new Error(path + ' ' + r.status);
  return r.json();
};

function fail(message) {
  let t = $('#err'); if (!t) { t = document.createElement('div'); t.id = 'err'; t.className = 'toast'; t.setAttribute('role', 'alert'); $('#wizard').prepend(t); }
  t.textContent = message; t.classList.remove('hidden'); clearTimeout(fail.timer); fail.timer = setTimeout(() => t.classList.add('hidden'), 8000);
}
const guard = (fn) => async (...a) => { try { return await fn(...a); } catch (e) { console.error(e); fail('Something went wrong talking to the server. Please try again.'); } };

/* ---------------------------------------------------------- plain-language words */
const QUAD = { hi_v_hi_a: 'upbeat and wired', hi_v_lo_a: 'upbeat and calm', lo_v_hi_a: 'low and wired', lo_v_lo_a: 'low and calm' };
const vWord = (v) => v > .5 ? 'high' : v > .2 ? 'good' : v < -.5 ? 'very low' : v < -.2 ? 'low' : 'even';
const aWord = (a) => a > .5 ? 'very wired' : a > .2 ? 'lively' : a < -.5 ? 'very calm' : a < -.2 ? 'calm' : 'steady';
const sureWord = (m) => { const sd = (m.sd_valence + m.sd_arousal) / 2; return sd < .6 ? 'fairly sure' : sd < .8 ? 'moderately sure' : 'still guessing'; };
const fitWord = (d) => d <= .25 ? 'a very close fit' : d <= .5 ? 'a close fit' : 'a looser fit, picked for variety';
const stratName = (s) => s === 'match' ? 'Match your mood' : 'A change of pace';
function fmtTime(ts) {
  const d = new Date(ts), now = new Date();
  const t = d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  return d.toDateString() === now.toDateString() ? t : d.toLocaleDateString([], { day: 'numeric', month: 'short' }) + ', ' + t;
}

/* ---------------------------------------------------------- mood plot (drawn into every .mood-plot) */
const moodCardHTML = () => `<aside class="card moodcard"><p class="eyebrow">What I think so far</p>
  <h3 class="mood-head">Not sure yet</h3>
  <svg class="mood-plot" viewBox="0 0 260 260" role="img" aria-label="Your estimated mood on a plot of pleasant versus unpleasant and calm versus intense"></svg>
  <p class="muted small mood-text"></p></aside>`;

function plotSVG(m) {
  const S = 260, P = 22, W = S - 2 * P, x = v => P + (v + 1) / 2 * W, y = a => P + (1 - (a + 1) / 2) * W;
  const r = Math.min(0.9, 0.67 * ((m.sd_valence + m.sd_arousal) / 2)) * W / 2;
  return `<rect x="${P}" y="${P}" width="${W}" height="${W}" fill="none" stroke="currentColor" stroke-opacity=".3"/>
    <line x1="${P}" y1="${S / 2}" x2="${S - P}" y2="${S / 2}" stroke="currentColor" stroke-opacity=".18"/>
    <line x1="${S / 2}" y1="${P}" x2="${S / 2}" y2="${S - P}" stroke="currentColor" stroke-opacity=".18"/>
    <g font-size="9" fill="currentColor" fill-opacity=".6">
      <text x="${P + 4}" y="${P + 11}">TENSE</text><text x="${S - P - 4}" y="${P + 11}" text-anchor="end">EXCITED</text>
      <text x="${P + 4}" y="${S - P - 5}">LOW</text><text x="${S - P - 4}" y="${S - P - 5}" text-anchor="end">CALM</text>
      <text x="${S / 2}" y="${S - 5}" text-anchor="middle">unpleasant &#8594; pleasant</text>
      <text x="9" y="${S / 2}" transform="rotate(-90 9 ${S / 2})" text-anchor="middle">calm &#8594; intense</text></g>
    <circle cx="${x(m.valence)}" cy="${y(m.arousal)}" r="${r}" fill="var(--warm)" fill-opacity=".16" stroke="var(--warm)" stroke-dasharray="3 3"/>
    <circle cx="${x(m.valence)}" cy="${y(m.arousal)}" r="5.5" fill="var(--warm)"/>`;
}
function drawPlot(m) {
  $$('svg.mood-plot').forEach(s => { s.innerHTML = plotSVG(m); });
  $$('.mood-head').forEach(h => { h.textContent = `Reads as ${vWord(m.valence)} mood and ${aWord(m.arousal)} energy`; });
  $$('.mood-text').forEach(t => { t.textContent = `I'm ${sureWord(m)}. The dashed ring is how unsure I am: the smaller it is, the better I know how you feel.`; });
}
const refreshMood = guard(async () => { state.mood = await api('/mood/' + state.user.user_id); drawPlot(state.mood); return state.mood; });

/* ---------------------------------------------------------- stepper */
function setStep(n, focus = true) {
  state.step = n; state.furthest = Math.max(state.furthest, n);
  for (let i = 1; i <= 4; i++) {
    const tab = $('#tab-' + i);
    tab.setAttribute('aria-selected', String(i === n)); tab.tabIndex = i === n ? 0 : -1;
    tab.disabled = i > state.furthest; tab.classList.toggle('done', state.done.has(i) && i !== n);
    $('#step-' + i).classList.toggle('hidden', i !== n);
  }
  if (n === 2 && !state.quizStarted) startQuiz(false);
  if (focus) {
    $('#wizard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    const h = $('#step-' + n + ' h2'); if (h) h.focus({ preventScroll: true });
  }
}
function next(from) { state.done.add(from); setStep(from + 1); }
$$('.steps [role=tab]').forEach(t => t.onclick = () => { const n = +t.dataset.step; if (n <= state.furthest) setStep(n); });

/* ---------------------------------------------------------- step 1: your day (typing rhythm) */
$('#free').addEventListener('keydown', (e) => {
  const t = performance.now(); const gap = state.lastKey === null ? 0 : t - state.lastKey; state.lastKey = t;
  const cls = e.key === 'Backspace' ? 'backspace' : e.key === 'Enter' ? 'enter' : e.key === ' ' ? 'space' : e.key.length === 1 ? 'char' : 'other';
  if (cls === 'other') return;
  state.events.push([gap, cls]);                         // timing and key class only; never the character
  const n = state.events.length;
  $('#free-status').textContent = n < 30 ? `${30 - n} more keystrokes and I can read your rhythm.` : 'That is enough for me. Continue whenever you like.';
});
async function submitTyping() {
  if (state.events.length < 30) return;
  const res = await api('/typing', { user_id: state.user.user_id, events: state.events });
  state.events = []; state.lastKey = null; $('#free').value = '';       // the text is discarded here
  $('#free-status').textContent = res.used ? 'Got it. The words stayed here.' : 'That was not enough to read. No problem.';
  await refreshMood();
}
$('#s1-next').onclick = guard(async () => { $('#s1-next').disabled = true; try { await submitTyping(); } finally { $('#s1-next').disabled = false; } next(1); });
$('#s1-skip').onclick = () => next(1);

/* ---------------------------------------------------------- step 2: your energy (adaptive questions) */
const quizArea = () => $('#quiz-area');
async function startQuiz(force) {
  state.quizStarted = true; state.answers = [];
  quizArea().innerHTML = '<p class="muted">Checking what I already know about you…</p>';
  try {
    const r = await api('/quiz/next', { user_id: state.user.user_id, answers: [], force });
    if (r.done) return quizSummary('known'); renderQuestion(r);
  } catch (e) { console.error(e); fail('Could not load the questions.'); quizArea().innerHTML = ''; state.quizStarted = false; }
}
function renderQuestion(r) {
  const n = state.answers.length + 1, sd = ((r.estimate.sd_valence + r.estimate.sd_arousal) / 2).toFixed(2);
  quizArea().innerHTML = `
    <div class="qmeta"><span class="dots" aria-hidden="true">${Array.from({ length: 6 }, (_, i) => `<i class="${i < n ? 'on' : ''}"></i>`).join('')}</span>
      <span class="muted small">Question ${n} (I stop as soon as I'm sure, at most 6)</span></div>
    <p class="qprompt" id="qprompt">${esc(r.item.prompt)}</p>
    <div class="opts" role="group" aria-labelledby="qprompt">${r.item.options.map((o, i) => `<button type="button" data-i="${i}">${esc(o)}</button>`).join('')}</div>
    <p class="muted small">I only ask because I'm not sure yet (doubt ${sd}). <button class="link" id="skipq" type="button">Skip the questions</button></p>`;
  state.quizShownAt = performance.now();
  $$('.opts button', quizArea()).forEach(b => b.onclick = guard(() => answer(r.item.item_id, +b.dataset.i)));
  $('#skipq').onclick = () => { state.answers = []; quizSummary('skipped'); };
  $('#quiz-title').textContent = 'A few quick questions';
}
async function answer(item_id, a) {
  state.answers.push({ item_id, answer: a, latency_ms: Math.round(performance.now() - state.quizShownAt) });
  const r = await api('/quiz/next', { user_id: state.user.user_id, answers: state.answers, force: true });
  drawPlot({ ...r.estimate });                                   // live preview: the ring shrinks as I learn
  if (r.done) { await api('/quiz/commit', { user_id: state.user.user_id, answers: state.answers }); const n = state.answers.length; state.answers = []; await refreshMood(); quizSummary('asked', n); }
  else renderQuestion(r);
}
function quizSummary(kind, n) {
  const m = state.mood; const read = m ? `${vWord(m.valence)} mood and ${aWord(m.arousal)} energy` : 'how you feel';
  const text = kind === 'known'
    ? `I already have a good read on you from earlier, so I don't need to ask anything right now. I have you down as <b>${read}</b>.`
    : kind === 'skipped' ? 'No problem, I will go with what I have. My guess may be rougher without your answers.'
    : `Thanks. I asked <b>${n} question${n === 1 ? '' : 's'}</b> and I read you as <b>${read}</b> (${m ? sureWord(m) : ''}).`;
  $('#quiz-title').textContent = kind === 'asked' ? 'Got it' : 'No questions needed';
  quizArea().innerHTML = `<div class="note-card"><p>${text}</p></div>
    <div class="row"><button class="primary" id="s2-next" type="button">Continue</button><button class="link" id="s2-again" type="button">${kind === 'asked' ? 'Redo the questions' : 'Ask me anyway'}</button></div>`;
  $('#s2-next').onclick = () => next(2);
  $('#s2-again').onclick = () => startQuiz(true);
}
$('#s2-back').onclick = () => setStep(1);

/* ---------------------------------------------------------- step 3: time, language, genre (and who is watching) */
const RUNTIMES = [['', 'Any length'], ['100', 'Under 100 minutes'], ['120', 'Under 2 hours'], ['150', 'Under 2½ hours']];
const LANGS = [['', 'Any language'], ['en', 'English'], ['hi', 'Hindi'], ['fr', 'French'], ['ja', 'Japanese'], ['ko', 'Korean'], ['es', 'Spanish']];
function radioChips(box, options, key) {
  box.innerHTML = '';
  options.forEach(([val, label]) => {
    const c = document.createElement('span'); c.className = 'chip'; c.textContent = label; c.setAttribute('role', 'radio'); c.tabIndex = 0;
    const sync = () => $$('.chip', box).forEach(x => x.setAttribute('aria-checked', String(x === c && state[key] === val)));
    const pick = () => { state[key] = val; $$('.chip', box).forEach(x => x.setAttribute('aria-checked', String(x === c))); };
    c.onclick = pick; c.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } };
    c.setAttribute('aria-checked', String(state[key] === val)); box.appendChild(c);
  });
}
function toggleChips(box, names, set, label = (x) => x) {
  box.innerHTML = '';
  names.forEach(name => {
    const c = document.createElement('span'); c.className = 'chip'; c.textContent = label(name); c.setAttribute('role', 'button'); c.tabIndex = 0; c.setAttribute('aria-pressed', String(set.has(name)));
    const t = () => { set.has(name) ? set.delete(name) : set.add(name); c.setAttribute('aria-pressed', String(set.has(name))); };
    c.onclick = t; c.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); t(); } };
    box.appendChild(c);
  });
}
const buildFilters = guard(async () => {
  radioChips($('#f-runtime'), RUNTIMES, 'runtime'); radioChips($('#f-lang'), LANGS, 'language');
  const genres = (await api('/genres')).filter(g => g !== 'IMAX'); toggleChips($('#f-genres'), genres, state.genres);
  const others = localProfiles().filter(p => p.user_id !== state.user.user_id);
  $('#group-box').classList.toggle('hidden', !others.length);
  toggleChips($('#group-chips'), others.map(p => p.user_id), state.group, (id) => (others.find(p => p.user_id === id) || {}).name);
});
$('#s3-back').onclick = () => setStep(2);

/* ---------------------------------------------------------- step 4: your picks */
const POSTER = 'https://image.tmdb.org/t/p/w342';
function posterHTML(m) {
  const initial = esc((m.title || '?').trim().charAt(0).toUpperCase());
  return `<div class="poster ${m.poster_path ? '' : 'nopic'}" data-initial="${initial}">${m.poster_path ? `<img src="${POSTER}${esc(m.poster_path)}" alt="Poster for ${esc(m.title)}" loading="lazy">` : ''}</div>`;
}
function guardPosters(root) { $$('.poster img', root).forEach(img => img.addEventListener('error', () => { img.parentElement.classList.add('nopic'); img.remove(); })); }
const metaLine = (m) => [(m.genres || []).join(' · '), m.runtime_min ? m.runtime_min + ' min' : ''].filter(Boolean).join(' · ');
function qualityReason(m) { return m.avg_rating ? `Well liked: ${m.avg_rating.toFixed(1)} out of 5 from ${m.n_ratings.toLocaleString()} MovieLens ratings.` : ''; }

function filmCardSingle(m) {
  const ex = m.explanation, reasons = [ex.why];
  if (ex.feel_tags && ex.feel_tags.length) reasons.push(`It feels ${ex.feel_tags.map(esc).join(', ')}.`);
  reasons.push(`Mood fit: ${fitWord(ex.affect_distance)} to the mood I aimed for.`);
  const q = qualityReason(m); if (q) reasons.push(q);
  if (ex.similar_mood_boost > 0) reasons.push('It worked well for other people who felt the way you do right now.');
  return `<article class="film">${posterHTML(m)}<div>
    <div class="fhead"><h3>${esc(m.title)}<span class="yr">${m.year ?? ''}</span></h3><span class="badge ${ex.strategy}">${ex.strategy === 'match' ? 'Matches you' : 'Change of pace'}</span></div>
    <p class="fmeta">${esc(metaLine(m))}</p>
    <p class="overview">${esc(m.overview || '')}</p>
    <div class="why"><b>Why this film</b><ul>${reasons.map(r => `<li>${r}</li>`).join('')}</ul></div>
    <div class="factions"><button type="button" data-rec="${m.rec_id}" data-title="${esc(m.title)}">I watched this</button></div></div></article>`;
}
function filmCardGroup(m, members) {
  const fits = Object.entries(m.member_distance).map(([n, d]) => `<li>For ${esc(n)}: ${fitWord(d)}.</li>`);
  const q = qualityReason(m);
  return `<article class="film">${posterHTML(m)}<div>
    <div class="fhead"><h3>${esc(m.title)}<span class="yr">${m.year ?? ''}</span></h3><span class="badge group">For the group</span></div>
    <p class="fmeta">${esc(metaLine(m))}</p>
    <p class="overview">${esc(m.overview || '')}</p>
    <div class="why"><b>Why this film</b><ul>${fits.join('')}<li>Picked so that nobody's mood is left far behind.</li>${q ? `<li>${q}</li>` : ''}</ul></div>
    <div class="factions">${Object.entries(m.rec_ids).map(([uid, rid]) => `<button type="button" data-rec="${rid}" data-uid="${uid}" data-title="${esc(m.title)}" data-who="${esc((members.find(x => x.user_id == uid) || {}).name || '')}">${esc((members.find(x => x.user_id == uid) || {}).name || 'Someone')} watched</button>`).join('')}</div></div></article>`;
}
function summarySingle(res) {
  const m = res.mood, seen = res.arm_obs[res.strategy], pct = Math.round(res.arm_means[res.strategy] * 100);
  const mw = `${vWord(m.valence)} mood and ${aWord(m.arousal)} energy`;
  const aim = `${vWord(res.target.valence)} mood and ${aWord(res.target.arousal)} energy`;
  const record = `In this kind of mood it has worked ${pct}% of the time for you (${seen} watch${seen === 1 ? '' : 'es'}).`;
  let why;
  if (res.strategy === 'regulate') {
    why = `You're feeling ${mw}. For most people a change of pace helps more than staying in the mood, so I aimed for ${aim}. ` +
      (seen > 0 ? record : "I haven't seen what works for you in this kind of mood yet.");
  } else {
    why = `You're feeling ${mw}. ` + (seen > 0
      ? `I'm matching that mood because, for you, it has worked ${pct}% of the time in this kind of mood (${seen} watch${seen === 1 ? '' : 'es'}).`
      : "This time I'm trying films that match that mood, so I can learn what works for you.");
  }
  return `<p class="eyebrow">Step 4</p><h2 tabindex="-1">Here's what I'd watch tonight</h2>
    <div class="sumrow"><div class="fact"><span class="k">Your mood</span><span class="v">${vWord(m.valence)}</span></div>
      <div class="fact"><span class="k">Your energy</span><span class="v">${aWord(m.arousal)}</span></div>
      <div class="fact"><span class="k">My approach</span><span class="v">${stratName(res.strategy)}</span></div></div>
    <p class="muted" style="margin:0">${why}</p>`;
}
function summaryGroup(res) {
  return `<p class="eyebrow">Step 4</p><h2 tabindex="-1">Films for the whole sofa</h2>
    <div class="sumrow">${res.members.map(mm => `<div class="fact"><span class="k">${esc(mm.name)}</span><span class="v">${vWord(mm.mood.valence)} mood, ${aWord(mm.mood.arousal)} energy</span><span class="muted small">${mm.strategy === 'match' ? 'wants to match' : 'a change of pace'}</span></div>`).join('')}</div>
    <p class="muted" style="margin:0">Each person's mood and what has worked for them both count. Whoever is feeling lowest gets a little extra say, and films are ranked so nobody is left far behind.</p>`;
}

function showResults(res, isGroup) {
  $('#res-summary').innerHTML = isGroup ? summaryGroup(res) : summarySingle(res);
  const box = $('#films'); const members = isGroup ? res.members : [];
  box.innerHTML = res.slate.length ? res.slate.map(m => isGroup ? filmCardGroup(m, members) : filmCardSingle(m)).join('')
    : '<div class="card"><p>I couldn\'t find films that match all of those filters. Try loosening one of them.</p></div>';
  guardPosters(box);
  $$('button[data-rec]', box).forEach(b => b.onclick = () => openAfter(+b.dataset.rec, b.dataset.title, b.dataset.who, b.dataset.uid ? +b.dataset.uid : null));
  $('#after').classList.add('hidden');
  state.done.add(3); setStep(4); loadAbout();
}
$('#find').onclick = guard(async () => {
  const btn = $('#find'); btn.disabled = true; btn.textContent = 'Finding films…';
  try {
    const filters = { max_runtime: +state.runtime || null, language: state.language || null, genres: state.genres.size ? [...state.genres] : null };
    if (state.group.size) {
      const members = [state.user, ...localProfiles().filter(p => state.group.has(p.user_id))].map(p => ({ user_id: p.user_id, token: p.token }));
      showResults(await api('/household/recommend', { members, ...filters }), true);
    } else showResults(await api('/recommend', { user_id: state.user.user_id, ...filters }), false);
  } finally { btn.disabled = false; btn.textContent = 'Show me films'; }
});
$('#s4-back').onclick = () => setStep(3);
$('#restart').onclick = () => {
  state.events = []; state.lastKey = null; state.answers = []; state.quizStarted = false; state.furthest = 1; state.done = new Set();
  $('#free').value = ''; $('#free-status').textContent = 'I need about 30 keystrokes to read your rhythm. It is fine to skip this.';
  $('#films').innerHTML = ''; $('#after').classList.add('hidden'); $('#about').innerHTML = '';
  setStep(1);
};

/* ---------------------------------------------------------- after watching */
function openAfter(rec_id, title, who, uid) {
  state.activeRec = rec_id; state.activeToken = uid ? tokenOf(uid) : state.user.token; state.activeWho = who || ''; state.liked = null;
  $('#after-title').textContent = who ? `How does ${who} feel after ${title}?` : `How do you feel after ${title}?`;
  $('#after-v').value = 0; $('#after-a').value = 0; $('#after-toast').classList.add('hidden');
  $('#liked-yes').style.borderColor = $('#liked-no').style.borderColor = '';
  $('#after').classList.remove('hidden'); $('#after').scrollIntoView({ behavior: 'smooth', block: 'center' }); $('#after-title').focus({ preventScroll: true });
}
$('#liked-yes').onclick = () => { state.liked = true; $('#liked-yes').style.borderColor = 'var(--cool)'; $('#liked-no').style.borderColor = ''; };
$('#liked-no').onclick = () => { state.liked = false; $('#liked-no').style.borderColor = 'var(--warm)'; $('#liked-yes').style.borderColor = ''; };
$('#after-submit').onclick = guard(async () => {
  const res = await api('/checkin', { rec_id: state.activeRec, valence_after: +$('#after-v').value / 100, arousal_after: +$('#after-a').value / 100, liked: state.liked }, state.activeToken);
  const who = state.activeWho;
  const t = $('#after-toast'); t.classList.remove('hidden');
  t.textContent = `Learned: ${res.strategy === 'match' ? 'matching the mood' : 'a change of pace'} scored ${Math.round(res.reward * 100)}% for ${who || 'you'} when ${who ? 'they were' : 'you were'} feeling ${QUAD[res.quadrant]}. It is now part of "What I've learned" below.`;
  await loadAbout();
});

/* ---------------------------------------------------------- About you: mood now, last readings, what I've learned */
function scaleHTML(label, lo, hi, v, word) {
  return `<div class="scale"><div class="lab"><span>${lo}</span><span class="val">${label}: ${word} (${v >= 0 ? '+' : ''}${v.toFixed(2)})</span><span>${hi}</span></div>
    <div class="track"><i style="left:${(v + 1) / 2 * 100}%"></i></div></div>`;
}
function trajSVG(t) {
  if (t.length < 2) return '<p class="muted small">Not enough readings yet for a chart. It fills in as you use the app.</p>';
  const x = i => 8 + i / (t.length - 1) * 284, y = v => 100 - (v + 1) / 2 * 90;
  const line = (k, col) => `<polyline fill="none" stroke="${col}" stroke-width="2" points="${t.map((p, i) => x(i) + ',' + y(p[k])).join(' ')}"/>`;
  return `<svg viewBox="0 0 300 116" role="img" aria-label="Your mood and energy over your recent readings" style="width:100%;height:auto">
    <line x1="8" y1="55" x2="292" y2="55" stroke="currentColor" stroke-opacity=".2"/>${line('valence', 'var(--cool)')}${line('arousal', 'var(--warm)')}
    <text x="10" y="112" font-size="9" fill="var(--cool)">mood</text><text x="50" y="112" font-size="9" fill="var(--warm)">energy</text></svg>`;
}
function readingHTML(r) {
  const mood = r.valence == null ? '' : `${vWord(r.valence)} mood`, en = r.arousal == null ? '' : `${aWord(r.arousal)} energy`;
  let text = '';
  if (r.source === 'quiz') text = `<b>Quick questions</b>${r.n_items ? ` (${r.n_items})` : ''}: ${mood}, ${en}.`;
  else if (r.source === 'checkin') text = `<b>After a film</b>: you said ${mood}, ${en}.`;
  else if (r.source === 'typing') {
    const pace = r.arousal > .1 ? 'a little quicker than your usual' : r.arousal < -.1 ? 'a little slower than your usual' : 'about your usual pace';
    const care = r.valence < -.08 ? ', with more corrections or pauses than usual' : r.valence > .08 ? ', and smoother than usual' : '';
    text = `<b>How you typed</b>: ${pace}${care}.`;
  } else text = `<b>Time of day</b>: only a small hint ${r.arousal > .05 ? 'toward livelier' : r.arousal < -.05 ? 'toward calmer' : '(neutral)'}.`;
  return `<li><span class="t">${esc(fmtTime(r.ts))}</span><span>${text}</span></li>`;
}
function learnedHTML(pol) {
  let total = 0, lines = '';
  const rows = Object.keys(pol).map(q => {
    const mt = pol[q].match, rg = pol[q].regulate; total += mt.observations + rg.observations;
    const cell = (a, color) => `<td><div class="bar"><i style="width:${Math.round(a.mean * 100)}%;background:${color}"></i></div><span class="n">${a.observations ? `${a.observations} watch${a.observations === 1 ? '' : 'es'}` : 'no watches yet'}</span></td>`;
    if (mt.observations + rg.observations > 0) {
      const better = mt.mean >= rg.mean ? ['matching', mt, rg] : ['a change of pace', rg, mt];
      lines += `<li>When you're ${QUAD[q]}, <b>${better[0]}</b> has worked better so far (${Math.round(better[1].mean * 100)}% against ${Math.round(better[2].mean * 100)}%).</li>`;
    }
    return `<tr><td>${QUAD[q]}</td>${cell(mt, 'var(--cool)')}${cell(rg, 'var(--warm)')}</tr>`;
  }).join('');
  const story = total === 0
    ? `<p>I haven't learned anything about you yet. After you watch a film, tell me how you feel and this fills in.</p>`
    : `<ul style="margin:6px 0 0;padding-left:18px">${lines}</ul>`;
  return `<table class="learn"><thead><tr><th>You feel</th><th>Match it</th><th>Change it</th></tr></thead><tbody>${rows}</tbody></table>${story}
    <div class="say2"><p><b>Why I know this.</b> Each time you check in after a film, I compare how you felt before and after (and whether you enjoyed it) and update these scores. Until there is enough evidence, I lean toward a change of pace, because it helps most people.</p></div>`;
}
const loadAbout = guard(async () => {
  const id = state.user.user_id;
  const [m, traj, reads, pol] = await Promise.all([api('/mood/' + id), api('/trajectory/' + id), api('/readings/' + id + '?limit=6'), api('/policy/' + id)]);
  state.mood = m;
  const last = reads[0] ? `Last reading: ${fmtTime(m.last_update || reads[0].ts)} (${{ quiz: 'quick questions', typing: 'how you typed', context: 'time of day', checkin: 'after a film' }[reads[0].source] || reads[0].source}).` : '';
  $('#about').innerHTML = `
    <section class="card"><p class="eyebrow">Your mood and energy</p>
      <h3 class="mood-head"></h3>
      <svg class="mood-plot" viewBox="0 0 260 260" role="img" aria-label="Your estimated mood"></svg>
      ${scaleHTML('Mood', 'Low', 'High', m.valence, vWord(m.valence))}${scaleHTML('Energy', 'Calm', 'Wired', m.arousal, aWord(m.arousal))}
      <p class="muted small mood-text"></p><p class="small">${esc(last)}</p></section>
    <section class="card"><p class="eyebrow">Your last readings</p>
      <h3>How I got here</h3>${trajSVG(traj)}
      <ul class="reads">${reads.map(readingHTML).join('') || '<li><span class="t"></span><span class="muted">No readings yet.</span></li>'}</ul></section>
    <section class="card"><p class="eyebrow">What I've learned about you</p>
      <h3>What has worked for you</h3>${learnedHTML(pol)}</section>`;
  drawPlot(m);
});

/* ---------------------------------------------------------- profiles and boot */
function loadProfiles() {
  const box = $('#profiles'); box.innerHTML = '';
  const users = localProfiles();
  users.forEach(u => { const b = document.createElement('button'); b.textContent = u.name; b.onclick = guard(() => enter(u)); box.appendChild(b); });
  if (!users.length) box.innerHTML = '<span class="muted small">No profiles on this device yet.</span>';
}
async function enter(u) {
  state.user = u; localStorage.setItem('reelstate_user', String(u.user_id));
  $('#screen-profile').classList.add('hidden'); $('#wizard').classList.remove('hidden');
  $('#who').innerHTML = `${esc(u.name)} <button class="link" id="switch" type="button">switch</button>`;
  $('#switch').onclick = () => { localStorage.removeItem('reelstate_user'); location.reload(); };
  $$('.mood-slot').forEach(s => { s.innerHTML = moodCardHTML(); });
  state.furthest = 1; state.done = new Set(); state.quizStarted = false;
  await api('/context/' + u.user_id, {});
  await buildFilters(); await refreshMood();
  setStep(1, false);
}
function bootTry() {
  loadProfiles();
  $('#new-profile').onsubmit = guard(async (e) => {
    e.preventDefault(); const name = $('#new-name').value.trim(); if (!name) return;
    const { user_id, token } = await api('/users', { name }, '');
    const profile = { user_id, name, token }; saveProfiles([...localProfiles(), profile]); await enter(profile);
  });
  const saved = +localStorage.getItem('reelstate_user'); const me = localProfiles().find(p => p.user_id === saved);
  if (me) enter(me).catch(() => localStorage.removeItem('reelstate_user'));
}
