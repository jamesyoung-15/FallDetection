const videoElement = document.getElementById('videoElement');

function handleCameraStream(stream) {
    videoElement.srcObject = stream;
}

function handleError(error) {
    console.error('Error accessing camera:', error);
}

function startCamera(deviceId) {
    const constraints = {
        video: { deviceId: { exact: deviceId } }
    };

    navigator.mediaDevices.getUserMedia(constraints)
        .then(handleCameraStream)
        .catch(handleError);
}

// Prompt user to select camera
function promptCameraSelection(devices) {
    const selectElement = document.createElement('select');
    devices.forEach(device => {
        if (device.kind === 'videoinput') {
            const optionElement = document.createElement('option');
            optionElement.value = device.deviceId;
            optionElement.textContent = device.label || `Camera ${selectElement.length + 1}`;
            selectElement.appendChild(optionElement);
        }
    });

    selectElement.addEventListener('change', event => {
        const selectedDeviceId = event.target.value;
        startCamera(selectedDeviceId);
    });

    document.body.appendChild(selectElement);
}

// Get the list of available devices
navigator.mediaDevices.enumerateDevices()
    .then(devices => {
        promptCameraSelection(devices);
    })
    .catch(handleError);