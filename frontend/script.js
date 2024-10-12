const trainForm = document.getElementById('train-form');
const predictForm = document.getElementById('predict-form');
const predictionResult = document.getElementById('prediction-result');

trainForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const xValues = document.getElementById('x-values').value.split(',').map(Number);
    const yValues = document.getElementById('y-values').value.split(',').map(Number);

    try {
        const response = await fetch('http://localhost:3000/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ X: xValues, y: yValues }),
            credentials: 'include', 
        });

        const result = await response.json();
        alert(result.message || 'Training successful!');
    } catch (error) {
        console.error('Error during training:', error);
        alert('Failed to train the model. Please try again.');
    }
});

predictForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const predictValue = Number(document.getElementById('predict-value').value);

    try {
        const response = await fetch('http://localhost:3000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ X: [predictValue] }),
            credentials: 'include', 
        });

        const result = await response.json();

        if (result.prediction) {
            predictionResult.innerText = `Prediction: ${result.prediction}`;
        } else {
            predictionResult.innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        console.error('Error during prediction:', error);
        predictionResult.innerText = 'Failed to make a prediction. Please try again.';
    }
});
