document.addEventListener('DOMContentLoaded', function () {
    // Dark mode 
    const toggleDarkModeButton = document.getElementById('toggle-dark-mode');

    if (localStorage.getItem('darkMode') === 'enabled') {
        document.body.classList.add('dark-mode');
    }

    toggleDarkModeButton.addEventListener('click', function () {
        document.body.classList.toggle('dark-mode');

        if (document.body.classList.contains('dark-mode')) {
            localStorage.setItem('darkMode', 'enabled');
        } else {
            localStorage.setItem('darkMode', 'disabled');
        }
    });

    // pie chart
    const resultContainer = document.getElementById('result-container');
    resultContainer.style.display = 'none';

    document.getElementById('predictForm').addEventListener('submit', async function (e) {
        e.preventDefault();  // Prevent the default form submission

        // Get the input news text from the form
        const newsText = document.getElementById('newsInput').value;

        // Check if input is not empty
        if (!newsText.trim()) {
            alert('Please enter the text of a news article.');
            return;
        }

        resultContainer.style.opacity = '0';
        resultContainer.style.display = 'flex';

        try {
            const response = await fetch('/bert-predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',  
                },
                body: JSON.stringify({ news: newsText }),  
            });

            const result = await response.json();

            if (result.error) {
                document.getElementById('result').innerHTML = `Error: ${result.error}`;
                return;
            }

            // Display the result
            resultContainer.style.opacity = '1';
            document.getElementById('result').innerHTML = `Prediction: ${result.prediction}<br>(Fake: ${result.fake_score.toFixed(2)}%, Real: ${result.real_score.toFixed(2)}%)`;

            updatePieChart(result.real_score, result.fake_score);

        } catch (error) {
            console.error('Error:', error);
            alert('Something went wrong. Please try again.');
        }
    });

    //update the pie chart progress based on real and fake score
    function updatePieChart(realScore, fakeScore) {
        const pieChart = document.querySelector('.pie-chart');
        const realAngle = (realScore / 100) * 360;

        pieChart.style.background = `conic-gradient(
            #28a745 0deg ${realAngle}deg,
            #dc3545 ${realAngle}deg 360deg
        )`;

        document.getElementById('score-percent').textContent = `${realScore.toFixed(1)}%`;
    }
});
