document.addEventListener('DOMContentLoaded', function () {
    // Dark mode toggle
    const toggleDarkModeButton = document.getElementById('toggle-dark-mode');

    // Check localStorage to see if dark mode was previously enabled
    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
    }

    toggleDarkModeButton.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');

        // Store the dark mode preference in localStorage
        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
        } else {
            localStorage.setItem('darkMode', 'disabled');
        }
    });

    // Hide the result and pie chart by default
    const resultContainer = document.getElementById('result-container');
    resultContainer.style.display = 'none';

    document.getElementById('predictForm').addEventListener('submit', async function (e) {
        e.preventDefault();

        // Get the input news text from the form
        const newsText = document.getElementById('newsInput').value;

        // Check if input is not empty
        if (!newsText.trim()) {
            alert('Please enter the text of a news article.');
            return;
        }

        // Show loading animation
        resultContainer.style.opacity = '0';
        resultContainer.style.display = 'flex';

        // Simulate a delay for better animation experience (optional)
        setTimeout(async function () {
            // Send the input text to the Flask backend using a POST request
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ news: newsText }),
                });

                const result = await response.json();

                // Handle any errors from the backend
                if (result.error) {
                    document.getElementById('result').innerHTML = `Error: ${result.error}`;
                    resultContainer.style.display = 'flex'; // Show error message container
                    return;
                }

                // Show the result container after successful submission
                resultContainer.style.opacity = '1';

                // Update the result text
                document.getElementById('result').innerHTML = `Prediction: ${result.prediction}<br>(Fake: ${result.fake_score.toFixed(2)}%, Real: ${result.real_score.toFixed(2)}%)`;


                // Update the pie chart with the real and fake score
                updatePieChart(result.real_score, result.fake_score);

            } catch (error) {
                console.error('Error:', error);
                alert('Something went wrong. Please try again.');
            }
        }, 500); // 0.5s delay to simulate smooth animation
    });

    // Function to update the pie chart progress based on real and fake score
    function updatePieChart(realScore, fakeScore) {
        const pieChart = document.querySelector('.pie-chart');
        const scorePercentElem = document.getElementById('score-percent');

        const realPercentage = realScore;
        const fakePercentage = fakeScore;

        const realAngle = (realPercentage / 100) * 360;
        const fakeAngle = 360 - realAngle;

        // Update the pie chart to show green for real, red for fake
        pieChart.style.background = `conic-gradient(
            #28a745 0deg ${realAngle}deg, 
            #dc3545 ${realAngle}deg 360deg
        )`; // Green for real, red for fake

        // Update the text inside the pie chart
        scorePercentElem.textContent = `${realPercentage.toFixed(1)}%`;
    }
});
