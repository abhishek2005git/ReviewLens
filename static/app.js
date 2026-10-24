const form = document.querySelector('#review-form');
const review = document.querySelector('#review');
const count = document.querySelector('#character-count');
const button = document.querySelector('#predict-button');
const error = document.querySelector('#error');
const result = document.querySelector('#result-content');
const examples = {
  positive: 'Absolutely loved this place! Delicious food, wonderful service, and a perfect evening. I cannot wait to come back.',
  mixed: 'The food was okay and the service was average. Nothing special, but not terrible either. An ordinary experience overall.',
  negative: 'Terrible experience. The food was cold and disgusting, the staff were rude, and the place was dirty. Never coming back.'
};
const descriptions = ['', 'A disappointing experience', 'Some room for improvement', 'Somewhere in the middle', 'A good experience', 'An excellent experience'];
let lastReview = null;
function updateInput() {
  count.textContent = `${review.value.length.toLocaleString('en-US')} / 5,000`;
  error.hidden = true;
  if (lastReview !== null && review.value.trim() !== lastReview) {
    result.innerHTML = '<div class="empty-symbol" aria-hidden="true">☆</div><h3>Your review has changed.</h3><p>Predict again to update the rating.</p>';
    lastReview = null;
  }
}
review.addEventListener('input', updateInput);
document.querySelectorAll('[data-example]').forEach(example => {
  example.addEventListener('click', () => {
    review.value = examples[example.dataset.example];
    updateInput();
    review.focus();
  });
});
form.addEventListener('submit', async event => {
  event.preventDefault();
  error.hidden = true;
  button.disabled = true;
  review.readOnly = true;
  document.querySelectorAll('[data-example]').forEach(example => { example.disabled = true; });
  button.textContent = 'Reading your review…';
  const submitted = review.value.trim();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch('/api/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ review: submitted }), signal: controller.signal
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Unable to predict this review. Please try again.');
    if (!Number.isInteger(data.rating) || data.rating < 1 || data.rating > 5) throw new Error('The server returned an invalid rating. Please try again.');
    const stars = '★'.repeat(data.rating) + '☆'.repeat(5 - data.rating);
    result.innerHTML = `<div class="rating-number">${data.rating}<span>/ 5</span></div><div class="filled-stars" aria-label="${data.rating} out of 5 stars">${stars}</div><h3>${descriptions[data.rating]}</h3><p>Based on the words in your review.<br>${data.mode === 'demo' ? 'An illustrative demo prediction.' : 'A model estimate, not a human rating.'}</p>`;
    lastReview = submitted;
  } catch (failure) {
    error.textContent = failure.name === 'AbortError' ? 'The prediction took too long. Please try again.' : failure instanceof TypeError ? 'Cannot reach the app. Check that the server is running and try again.' : failure.message;
    error.hidden = false;
  } finally {
    clearTimeout(timeout);
    button.disabled = false;
    review.readOnly = false;
    document.querySelectorAll('[data-example]').forEach(example => { example.disabled = false; });
    button.innerHTML = 'Predict rating <span aria-hidden="true">↗</span>';
  }
});
