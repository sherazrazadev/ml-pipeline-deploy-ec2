document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('submitBtn');
    const loader = document.getElementById('loader');
    const btnText = submitBtn.querySelector('span');
    const resultCard = document.getElementById('result');

    // UI Loading State
    submitBtn.disabled = true;
    loader.style.display = 'block';
    btnText.style.opacity = '0.5';
    resultCard.classList.add('hidden');

    // Gather data
    const data = {
        sepal_length: parseFloat(document.getElementById('sepal_length').value),
        sepal_width: parseFloat(document.getElementById('sepal_width').value),
        petal_length: parseFloat(document.getElementById('petal_length').value),
        petal_width: parseFloat(document.getElementById('petal_width').value)
    };

    try {
        // Send request to backend
        // Note: In Docker Compose, we might need to adjust the URL if accessing from browser vs container.
        // For browser access, localhost:7001 is correct if mapped.
        const response = await fetch('http://localhost:7001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const result = await response.json();

        // Update UI
        document.getElementById('speciesName').textContent = result.class_name;
        document.getElementById('classId').textContent = result.class_id;
        resultCard.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to get prediction. Please check if the backend is running.');
    } finally {
        // Reset UI State
        submitBtn.disabled = false;
        loader.style.display = 'none';
        btnText.style.opacity = '1';
    }
});
