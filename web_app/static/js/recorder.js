// Dark Mode Toggle
const themeToggle = document.getElementById('themeToggle');
const htmlElement = document.documentElement;

// Check for saved theme preference or default to 'light'
const currentTheme = localStorage.getItem('theme') || 'light';
htmlElement.setAttribute('data-theme', currentTheme);

// Update icon based on current theme
updateThemeIcon(currentTheme);

themeToggle.addEventListener('click', () => {
    const currentTheme = htmlElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    
    htmlElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
    
    // Add rotation animation
    themeToggle.style.transform = 'rotate(360deg)';
    setTimeout(() => {
        themeToggle.style.transform = 'rotate(0deg)';
    }, 300);
});

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    if (theme === 'dark') {
        icon.className = 'fas fa-sun';
    } else {
        icon.className = 'fas fa-moon';
    }
}

// Audio Recording Functionality
let mediaRecorder;
let audioChunks = [];
let audioBlob;
let timerInterval;
let seconds = 0;

const micButton = document.getElementById('micButton');
const stopBtn = document.getElementById('stopBtn');
const analyzeBtn = document.getElementById('analyzeBtn');
const playBtn = document.getElementById('playBtn');
const audioPlayback = document.getElementById('audioPlayback');
const statusMessage = document.getElementById('statusMessage');
const resultsContainer = document.getElementById('resultsContainer');
const timerDisplay = document.getElementById('timerDisplay');
const timerText = document.getElementById('timerText');
const audioPlayerContainer = document.getElementById('audioPlayerContainer');

// Step indicators
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

// Microphone button click
micButton.addEventListener('click', async () => {
    if (micButton.classList.contains('recording')) {
        stopRecording();
    } else {
        await startRecording();
    }
});

// Stop button
stopBtn.addEventListener('click', () => {
    stopRecording();
});

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        audioChunks = [];
        seconds = 0;
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            const audioUrl = URL.createObjectURL(audioBlob);
            audioPlayback.src = audioUrl;
            
            // Show audio player
            audioPlayerContainer.style.display = 'block';
            
            showStatus('Recording saved! Click "Analyze Now" to detect your accent.', 'ready');
            
            // Update step
            step1.querySelector('.step-circle').classList.add('completed');
            step1.querySelector('.step-circle').innerHTML = '<i class="fas fa-check"></i>';
        };
        
        mediaRecorder.start();
        
        // UI updates
        micButton.classList.add('recording');
        micButton.innerHTML = '<i class="fas fa-stop"></i>';
        stopBtn.style.display = 'block';
        timerDisplay.style.display = 'flex';
        resultsContainer.style.display = 'none';
        audioPlayerContainer.style.display = 'none';
        
        // Start timer
        timerInterval = setInterval(() => {
            seconds++;
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            timerText.textContent = `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        }, 1000);
        
        showStatus('Recording... Speak naturally in English', 'recording');
        
    } catch (error) {
        showStatus('Error accessing microphone: ' + error.message, 'recording');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
        
        // UI updates
        micButton.classList.remove('recording');
        micButton.innerHTML = '<i class="fas fa-microphone"></i>';
        stopBtn.style.display = 'none';
        timerDisplay.style.display = 'none';
        
        clearInterval(timerInterval);
    }
}

// Play button
playBtn.addEventListener('click', () => {
    if (audioPlayback.paused) {
        audioPlayback.play();
        playBtn.innerHTML = '<i class="fas fa-pause"></i>';
    } else {
        audioPlayback.pause();
        playBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
});

audioPlayback.addEventListener('ended', () => {
    playBtn.innerHTML = '<i class="fas fa-play"></i>';
});

// Analyze button
analyzeBtn.addEventListener('click', async () => {
    const formData = new FormData();
    formData.append('audio', audioBlob, 'recording.wav');
    
    showStatus('Analyzing your accent with AI...', 'processing');
    analyzeBtn.disabled = true;
    
    // Update step
    step2.querySelector('.step-circle').classList.add('active');
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data);
        } else {
            throw new Error(data.error || 'Prediction failed');
        }
        
    } catch (error) {
        showStatus('Error: ' + error.message, 'recording');
        analyzeBtn.disabled = false;
        step2.querySelector('.step-circle').classList.remove('active');
    }
});

function displayResults(data) {
    // Update step indicators
    step2.querySelector('.step-circle').classList.remove('active');
    step2.querySelector('.step-circle').classList.add('completed');
    step2.querySelector('.step-circle').innerHTML = '<i class="fas fa-check"></i>';
    step3.querySelector('.step-circle').classList.add('active');
    
    // Display accent details
    document.getElementById('regionValue').textContent = data.region;
    document.getElementById('languageValue').textContent = data.native_language;
    document.getElementById('confidenceValue').textContent = data.confidence + '%';
    document.getElementById('confidencePercent').textContent = data.confidence + '%';
    document.getElementById('modelBadge').textContent = `${data.model_info.model} (Layer ${data.model_info.layer})`;
    document.getElementById('accentHighlight').textContent = data.region + ' accent';
    
    // Animate confidence meter
    setTimeout(() => {
        document.getElementById('meterFill').style.width = data.confidence + '%';
    }, 100);
    
    // Display cuisines
    const cuisinesGrid = document.getElementById('cuisinesGrid');
    cuisinesGrid.innerHTML = '';
    
    data.cuisines.forEach((cuisine, index) => {
        setTimeout(() => {
            const cuisineCard = document.createElement('div');
            cuisineCard.className = 'cuisine-item';
            cuisineCard.innerHTML = `
                <h3>${cuisine.name}</h3>
                <p>${cuisine.description}</p>
            `;
            cuisinesGrid.appendChild(cuisineCard);
        }, index * 150);
    });

    // Show results
    resultsContainer.style.display = 'block';
    showStatus('Analysis complete!', 'ready');
    
    // Scroll to results
    setTimeout(() => {
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 500);
}

function showStatus(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message show ${type}`;
}
