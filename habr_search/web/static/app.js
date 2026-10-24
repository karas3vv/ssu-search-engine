// элементы страницы не хранят ключ модели и не отправляют поиск при вводе текста
const form = document.getElementById('search-form');
const query = /** @type {HTMLInputElement} */ (document.getElementById('query'));
const searchButton = /** @type {HTMLButtonElement} */ (document.getElementById('search-button'));
const message = /** @type {HTMLElement} */ (document.getElementById('message'));
const suggestion = /** @type {HTMLElement} */ (document.getElementById('suggestion'));
const results = /** @type {HTMLOListElement} */ (document.getElementById('results'));
const section = /** @type {HTMLElement} */ (document.getElementById('results-section'));
const showMore = /** @type {HTMLButtonElement} */ (document.getElementById('show-more'));
const PAGE_SIZE = 5;
const RESULT_LIMIT = 20;
/** @typedef {{article_id:number,title:string,url:string,snippet:string,score:number}} Article */
/** @type {Article[]} */
let foundArticles = [];
let shownCount = 0;
let activeQuery = '';
let pending = false;
let searchGeneration = 0;

// примеры меняются только в пустом неактивном поле и не запускают поиск
const searchHints = [
  'Как создавать отчёты Word из Excel?',
  'Как сохранить страницу в PDF?',
  'Почему больше ядер не всегда быстрее?',
  'Как настроить домашнюю телефонию?',
  'Как менять яркость монитора автоматически?',
  'Как добавить cookie-баннер без плагина?',
];
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

// выбираем новую подсказку, отличную от текущей
function changeSearchHint() {
  const available = searchHints.filter(text => text !== query.placeholder);
  query.placeholder = available[Math.floor(Math.random() * available.length)];
}

changeSearchHint();
window.setInterval(() => {
  if (query.value || document.activeElement === query || document.hidden || pending || reducedMotion.matches) return;
  changeSearchHint();
}, 6000);

// меняем сообщение и его оформление без вставки html
/** @param {string} text @param {string} [kind] */
function status(text, kind = '') { message.textContent = text; message.className = `message ${kind}`; }

// выбираем русское окончание для числа статей
/** @param {number} count @param {string[]} forms */
function plural(count, forms) {
  const rest = count % 100;
  if (rest >= 11 && rest <= 14) return forms[2];
  const last = count % 10;
  return last === 1 ? forms[0] : last >= 2 && last <= 4 ? forms[1] : forms[2];
}

// отправляем запрос к своему api и переводим ошибки в короткие сообщения страницы
/** @param {string} path @param {object | undefined} [body] @returns {Promise<any>} */
async function api(path, body) {
  let response;
  try {
    response = await fetch(path, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
  } catch { throw new Error('Не удалось выполнить поиск. Попробуйте ещё раз.'); }
  let value;
  try { value = await response.json(); }
  catch { throw new Error('Не удалось получить результаты. Попробуйте ещё раз.'); }
  if (!response.ok) {
    if (response.status === 429) throw new Error('Поиск уже выполняется. Дождитесь результата.');
    if (response.status === 422) throw new Error('Проверьте текст запроса.');
    if (response.status === 400 && typeof value.detail === 'string' && value.detail.includes('токенов')) {
      throw new Error('Запрос слишком длинный. Попробуйте сократить его.');
    }
    throw new Error('Поиск временно недоступен. Попробуйте позже.');
  }
  return value;
}

// текст статьи вставляется как текст, а не как разметка из внешнего источника
/** @param {Article} item @returns {HTMLLIElement} */
function article(item) {
  const card = document.createElement('li'); card.className = 'result';
  const heading = document.createElement('h3');
  const link = document.createElement('a'); link.textContent = item.title;
  try {
    const url = new URL(item.url);
    if (url.protocol === 'https:' && url.hostname === 'habr.com') {
      link.href = url.href; link.target = '_blank'; link.rel = 'noopener noreferrer';
    }
  } catch { /* некорректный адрес остаётся обычным текстом */ }
  heading.append(link);
  const snippet = document.createElement('p'); snippet.textContent = item.snippet;
  card.append(heading, snippet); return card;
}

// при изменении поля не продолжаем выдачу другой поисковой фразы
function updateShowMore() {
  showMore.hidden = pending || query.value.trim() !== activeQuery || shownCount >= foundArticles.length;
}

// добавляем следующую пятёрку из готового ответа; повторного запроса к api здесь нет
/** @returns {HTMLLIElement | null} */
function appendResults() {
  const cards = foundArticles.slice(shownCount, shownCount + PAGE_SIZE).map(article);
  results.append(...cards);
  shownCount += cards.length;
  updateShowMore();
  return cards[0] || null;
}

showMore.addEventListener('click', () => {
  if (pending || query.value.trim() !== activeQuery) return;
  const firstNew = appendResults();
  // переносим фокус к новой статье, в том числе когда последняя кнопка уже скрылась
  firstNew?.querySelector('a')?.focus({preventScroll: true});
  firstNew?.scrollIntoView({behavior: reducedMotion.matches ? 'auto' : 'smooth', block: 'start'});
});

query.addEventListener('input', () => { ++searchGeneration; suggestion.hidden = true; updateShowMore(); });
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
  query.value = button.getAttribute('data-query') || ''; query.dispatchEvent(new Event('input')); query.focus();
}));

// исправление становится новым запросом только после нажатия на предложение
/** @param {string} originalQuery @param {string | null} suggestedQuery */
function showSuggestion(originalQuery, suggestedQuery) {
  suggestion.replaceChildren();
  suggestion.hidden = !suggestedQuery || query.value.trim() !== originalQuery.trim();
  if (suggestion.hidden) return;
  suggestion.append(document.createTextNode('Возможно, вы имели в виду: '));
  const button = document.createElement('button');
  button.type = 'button'; button.className = 'text-button spelling-query';
  button.textContent = suggestedQuery;
  button.addEventListener('click', () => {
    if (pending) return;
    query.value = suggestedQuery || originalQuery;
    query.dispatchEvent(new Event('input'));
    void runSearch(query.value);
  });
  suggestion.append(button);
}

// подсказка приходит после результатов и не может заменить подсказку нового запроса
/** @param {string} text @param {number} generation */
async function deferredSuggestion(text, generation) {
  try {
    const value = await api('/api/suggest', {query: text});
    if (generation !== searchGeneration || query.value.trim() !== text.trim()) return;
    showSuggestion(text, value.available && value.corrected_query !== text.trim() ? value.corrected_query : null);
  } catch { /* выдача остаётся доступна даже при сбое подсказок */ }
}

// каждый поиск использует введённый текст; исправление требует отдельного нажатия
/** @param {string} text */
async function runSearch(text) {
  if (pending) return;
  if (!text.trim()) { query.focus(); return; }
  const generation = ++searchGeneration;
  pending = true; searchButton.disabled = true; results.setAttribute('aria-busy', 'true');
  foundArticles = []; shownCount = 0; activeQuery = '';
  updateShowMore();
  section.hidden = true; suggestion.hidden = true;
  status('Ищем статьи…', 'loading');
  try {
    // сервер возвращает до двадцати кандидатов, сначала показываем только пять
    const value = await api('/api/search', {query: text, top_k: RESULT_LIMIT, correct_typos: false});
    foundArticles = value.results; activeQuery = text.trim();
    results.replaceChildren(); appendResults(); section.hidden = false;
    showSuggestion(text, value.suggested_query || null);
    if (value.deferred_suggestion) void deferredSuggestion(text, generation);
    status(value.results.length ? '' : 'Подходящих статей не найдено. Попробуйте уточнить запрос.');
  } catch (error) {
    status(error instanceof Error ? error.message : 'Не удалось выполнить поиск. Проверьте соединение и повторите запрос.', 'error');
  } finally {
    pending = false; searchButton.disabled = false; results.removeAttribute('aria-busy');
    updateShowMore();
  }
}

form?.addEventListener('submit', event => {
  event.preventDefault(); void runSearch(query.value.trim());
});

// показываем число статей при открытии страницы без обращения к модели
/** @returns {Promise<void>} */
async function updateHealth() {
  const value = await api('/api/health');
  const count = /** @type {HTMLElement} */ (document.getElementById('article-count'));
  count.textContent = value.articles ? `${value.articles} ${plural(value.articles, ['статья', 'статьи', 'статей'])}` : '';
  count.hidden = !value.articles;
}

updateHealth().catch(() => { /* ошибка будет показана при попытке поиска */ });
