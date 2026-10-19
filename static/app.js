// элементы страницы не хранят ключ модели и не отправляют поиск при вводе текста
const form = document.getElementById('search-form');
const query = /** @type {HTMLInputElement} */ (document.getElementById('query'));
const searchButton = /** @type {HTMLButtonElement} */ (document.getElementById('search-button'));
const message = /** @type {HTMLElement} */ (document.getElementById('message'));
const suggestion = /** @type {HTMLElement} */ (document.getElementById('suggestion'));
const results = /** @type {HTMLOListElement} */ (document.getElementById('results'));
const section = /** @type {HTMLElement} */ (document.getElementById('results-section'));
const usedQuery = /** @type {HTMLElement} */ (document.getElementById('used-query'));
let pending = false;

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

function changeSearchHint() {
  const available = searchHints.filter(text => text !== query.placeholder);
  query.placeholder = available[Math.floor(Math.random() * available.length)];
}

changeSearchHint();
window.setInterval(() => {
  if (query.value || document.activeElement === query || document.hidden || pending || reducedMotion.matches) return;
  changeSearchHint();
}, 6000);

/** @param {string} text @param {string} [kind] */
function status(text, kind = '') { message.textContent = text; message.className = `message ${kind}`; }

/** @param {number} count @param {string[]} forms */
function plural(count, forms) {
  const rest = count % 100;
  if (rest >= 11 && rest <= 14) return forms[2];
  const last = count % 10;
  return last === 1 ? forms[0] : last >= 2 && last <= 4 ? forms[1] : forms[2];
}

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
/** @param {{article_id:number,title:string,url:string,snippet:string,score:number}} item */
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

query.addEventListener('input', () => { suggestion.hidden = true; });
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
  query.value = button.getAttribute('data-query') || ''; query.dispatchEvent(new Event('input')); query.focus();
}));

/** @param {Record<string, string[]>} alternatives */
function showAlternatives(alternatives) {
  const entries = Object.entries(alternatives);
  suggestion.replaceChildren();
  suggestion.hidden = entries.length === 0;
  if (entries.length) {
    suggestion.textContent = entries.map(([word, options]) => `Варианты написания «${word}»: ${options.join(', ')}.`).join(' ');
  }
}

// исходная фраза захватывается из ответа, чтобы отмена исправления не зависела от новых правок поля
/** @param {string} text @param {boolean} [applyCorrections] */
async function runSearch(text, applyCorrections = true) {
  if (pending) return;
  if (!text.trim()) { query.focus(); return; }
  pending = true; searchButton.disabled = true; results.setAttribute('aria-busy', 'true');
  section.hidden = true; suggestion.hidden = true;
  status('Ищем статьи…', 'loading');
  try {
    // проверка доступности словаря локальная и не вызывает модель
    const available = applyCorrections ? await updateHealth() : false;
    const value = await api('/api/search', {query: text, top_k: 5, correct_typos: available});
    results.replaceChildren(...value.results.map(article)); section.hidden = false;
    usedQuery.replaceChildren(); usedQuery.hidden = !value.corrected;
    if (value.corrected) {
      usedQuery.append(document.createTextNode(`Показаны результаты для: ${value.used_query} `));
      const original = document.createElement('button');
      original.type = 'button'; original.className = 'text-button original-query';
      original.textContent = 'Искать исходный запрос';
      original.title = value.original_query;
      original.addEventListener('click', () => {
        query.value = value.original_query;
        query.dispatchEvent(new Event('input'));
        void runSearch(value.original_query, false);
      });
      usedQuery.append(original);
    }
    showAlternatives(value.alternatives || {});
    status(value.results.length ? '' : 'Подходящих статей не найдено. Попробуйте уточнить запрос.');
  } catch (error) {
    status(error instanceof Error ? error.message : 'Не удалось выполнить поиск. Проверьте соединение и повторите запрос.', 'error');
  } finally { pending = false; searchButton.disabled = false; results.removeAttribute('aria-busy'); }
}

form?.addEventListener('submit', event => {
  event.preventDefault(); void runSearch(query.value.trim());
});

/** @returns {Promise<boolean>} */
async function updateHealth() {
  const value = await api('/api/health');
  const count = /** @type {HTMLElement} */ (document.getElementById('article-count'));
  count.textContent = value.articles ? `${value.articles} ${plural(value.articles, ['статья', 'статьи', 'статей'])}` : '';
  count.hidden = !value.articles;
  return value.spelling_available;
}

updateHealth().catch(() => { /* ошибка будет показана при попытке поиска */ });
