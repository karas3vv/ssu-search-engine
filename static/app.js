// элементы страницы не хранят ключ модели и не отправляют поиск при вводе текста
const form = document.getElementById('search-form');
const query = /** @type {HTMLInputElement} */ (document.getElementById('query'));
const searchButton = /** @type {HTMLButtonElement} */ (document.getElementById('search-button'));
const checkButton = /** @type {HTMLButtonElement} */ (document.getElementById('check-spelling'));
const message = /** @type {HTMLElement} */ (document.getElementById('message'));
const suggestion = /** @type {HTMLElement} */ (document.getElementById('suggestion'));
const results = /** @type {HTMLOListElement} */ (document.getElementById('results'));
const section = /** @type {HTMLElement} */ (document.getElementById('results-section'));
const usedQuery = /** @type {HTMLElement} */ (document.getElementById('used-query'));
let pending = false;
let suggestionVersion = 0;

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
  } catch { throw new Error('Не удалось связаться с сервером поиска. Проверьте, запущен ли он.'); }
  let value;
  try { value = await response.json(); }
  catch { throw new Error('Сервер вернул неожиданный ответ. Проверьте терминал сервера.'); }
  if (!response.ok) throw new Error(typeof value.detail === 'string' ? value.detail : 'Проверьте текст и параметры запроса.');
  return value;
}

// текст статьи вставляется как текст, а не как разметка из внешнего источника
/** @param {{article_id:number,title:string,url:string,snippet:string,score:number}} item @param {number} index */
function article(item, index) {
  const card = document.createElement('li'); card.className = 'result';
  const top = document.createElement('div'); top.className = 'result-top';
  const number = document.createElement('span'); number.className = 'result-number'; number.textContent = String(index + 1).padStart(2, '0');
  const source = document.createElement('span'); source.textContent = 'ХАБР · СТАТЬЯ';
  const score = document.createElement('span'); score.className = 'score'; score.textContent = `сходство ${item.score.toFixed(4)}`;
  top.append(number, source, score);
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
  card.append(top, heading, snippet); return card;
}

query.addEventListener('input', () => { suggestionVersion++; suggestion.hidden = true; });
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
  query.value = button.getAttribute('data-query') || ''; query.dispatchEvent(new Event('input')); query.focus();
}));

checkButton.addEventListener('click', async () => {
  const text = query.value.trim();
  if (!text) { query.focus(); return; }
  const version = ++suggestionVersion; checkButton.disabled = true;
  try {
    const value = await api('/api/suggest', {query: text});
    if (version !== suggestionVersion) return;
    suggestion.replaceChildren(); suggestion.hidden = false;
    if (!value.available) { suggestion.textContent = 'Словарь опечаток не загружен.'; return; }
    if (value.corrected_query !== value.original_query) {
      suggestion.append(document.createTextNode('Возможно, вы имели в виду: '));
      const apply = document.createElement('button'); apply.type = 'button'; apply.textContent = value.corrected_query;
      apply.addEventListener('click', () => { query.value = value.corrected_query; query.dispatchEvent(new Event('input')); query.focus(); });
      suggestion.append(apply);
    } else {
      const entries = Object.entries(value.alternatives);
      suggestion.textContent = entries.length ? `Варианты для проверки: ${entries.map(([word, options]) => `${word} → ${/** @type {string[]} */ (options).join(', ')}`).join('; ')}. Запрос сохранён.` : 'Однозначных исправлений не найдено. Запрос сохранён.';
    }
  } catch (error) { status(error instanceof Error ? error.message : 'Не удалось проверить запрос.', 'error'); }
  finally { checkButton.disabled = false; }
});

/** @param {Record<string, string[]>} alternatives */
function showAlternatives(alternatives) {
  const entries = Object.entries(alternatives);
  suggestion.replaceChildren();
  suggestion.hidden = entries.length === 0;
  if (entries.length) {
    suggestion.textContent = `Варианты для проверки: ${entries.map(([word, options]) => `${word} → ${options.join(', ')}`).join('; ')}. Эти замены не применены.`;
  }
}

// исходная фраза захватывается из ответа, чтобы отмена исправления не зависела от новых правок поля
/** @param {string} text @param {boolean} [applyCorrections] */
async function runSearch(text, applyCorrections = true) {
  if (pending) return;
  if (!text.trim()) { query.focus(); return; }
  pending = true; searchButton.disabled = true; results.setAttribute('aria-busy', 'true');
  section.hidden = true; suggestion.hidden = true; suggestionVersion++;
  /** @type {HTMLElement} */ (document.getElementById('empty-state')).hidden = true;
  status('Ищем подходящие статьи. Ответ модели может занять некоторое время…', 'loading');
  try {
    // проверка доступности словаря локальная и не вызывает модель
    const available = applyCorrections ? await updateHealth() : false;
    const value = await api('/api/search', {query: text, top_k: 5, correct_typos: available});
    results.replaceChildren(...value.results.map(article)); section.hidden = false;
    /** @type {HTMLElement} */ (document.getElementById('result-count')).textContent = `${value.results.length} ${plural(value.results.length, ['статья', 'статьи', 'статей'])} · ${(value.elapsed_ms / 1000).toFixed(1)} с`;
    usedQuery.replaceChildren(document.createTextNode(`${value.corrected ? 'Показаны результаты для: ' : 'По запросу: '}${value.used_query}`));
    if (value.corrected) {
      const original = document.createElement('button');
      original.type = 'button'; original.className = 'text-button original-query';
      original.textContent = 'Искать исходный запрос';
      original.title = value.original_query;
      original.addEventListener('click', () => {
        query.value = value.original_query;
        query.dispatchEvent(new Event('input'));
        void runSearch(value.original_query, false);
      });
      usedQuery.append(document.createTextNode(' '), original);
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
  /** @type {HTMLElement} */ (document.getElementById('index-status')).textContent = `${value.articles ? `${value.articles} ${plural(value.articles, ['статья', 'статьи', 'статей'])} · ` : ''}${value.fragments} ${plural(value.fragments, ['фрагмент', 'фрагмента', 'фрагментов'])}`;
  checkButton.disabled = !value.spelling_available;
  /** @type {HTMLElement} */ (document.getElementById('spelling-note')).textContent = value.spelling_available ? 'Опечатки проверяются автоматически' : 'Поиск без исправлений: словарь не загружен';
  return value.spelling_available;
}

updateHealth().catch(() => { /** @type {HTMLElement} */ (document.getElementById('index-status')).textContent = 'Индекс недоступен'; });
