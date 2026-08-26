const recordBtn = document.getElementById("recordBtn");
const timerEl = document.getElementById("timer");
const statusEl = document.getElementById("status");
const fileInput = document.getElementById("fileInput");

const board = document.getElementById("board");
const boardLabel = document.getElementById("boardLabel");
const boardConfidence = document.getElementById("boardConfidence");
const boardBreakdown = document.getElementById("boardBreakdown");

const TARGET_SAMPLE_RATE = 16000;

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;

let recordedChunks = [];
let isRecording = false;

let recordStartTime = null;
let timerInterval = null;


// ======================================================
// RECORD BUTTON
// ======================================================

recordBtn.addEventListener("click", () => {

    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }

});


// ======================================================
// FILE UPLOAD
// ======================================================

fileInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    setStatus(`Uploaded: ${file.name}`);

    setBoardIdle();

    sendAudioBlob(file, file.name);

});


// ======================================================
// START RECORDING
// ======================================================

async function startRecording() {

    try {

        mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: true
        });

    } catch (error) {

        console.error("Microphone error:", error);

        setStatus(
            "Microphone access denied or unavailable."
        );

        return;
    }


    try {

        audioContext = new (
            window.AudioContext ||
            window.webkitAudioContext
        )();

        sourceNode =
            audioContext.createMediaStreamSource(
                mediaStream
            );

        processorNode =
            audioContext.createScriptProcessor(
                4096,
                1,
                1
            );

        recordedChunks = [];


        processorNode.onaudioprocess = (event) => {

            if (!isRecording) {
                return;
            }

            const input =
                event.inputBuffer.getChannelData(0);

            recordedChunks.push(
                new Float32Array(input)
            );

        };


        sourceNode.connect(processorNode);

        processorNode.connect(
            audioContext.destination
        );


        isRecording = true;

        recordStartTime = Date.now();


        recordBtn.textContent =
            "■ Stop Recording";

        recordBtn.classList.add(
            "recording"
        );


        setStatus(
            "Recording... speak now."
        );

        setBoardIdle();


        timerInterval = setInterval(() => {

            const seconds =
                (
                    (Date.now() - recordStartTime)
                    / 1000
                ).toFixed(1);

            timerEl.textContent =
                `${seconds}s`;

        }, 100);


    } catch (error) {

        console.error(
            "Audio initialization error:",
            error
        );

        setStatus(
            "Could not start audio recording."
        );

    }

}


// ======================================================
// STOP RECORDING
// ======================================================

function stopRecording() {

    if (!isRecording) {
        return;
    }


    isRecording = false;


    clearInterval(timerInterval);

    timerInterval = null;

    timerEl.textContent = "";


    recordBtn.textContent =
        "● Start Recording";

    recordBtn.classList.remove(
        "recording"
    );


    // Disconnect audio nodes safely

    if (processorNode) {

        processorNode.disconnect();

        processorNode.onaudioprocess = null;

    }


    if (sourceNode) {

        sourceNode.disconnect();

    }


    // Stop microphone

    if (mediaStream) {

        mediaStream
            .getTracks()
            .forEach(track => track.stop());

    }


    if (!audioContext) {

        setStatus(
            "Audio recording failed."
        );

        return;
    }


    const inputSampleRate =
        audioContext.sampleRate;


    // Combine recorded chunks

    const merged =
        mergeBuffers(recordedChunks);


    // Convert to 16 kHz

    const downsampled =
        downsampleBuffer(
            merged,
            inputSampleRate,
            TARGET_SAMPLE_RATE
        );


    // Convert PCM audio to WAV

    const wavBlob =
        encodeWAV(
            downsampled,
            TARGET_SAMPLE_RATE
        );


    // Close audio context

    audioContext.close();


    audioContext = null;
    sourceNode = null;
    processorNode = null;


    recordedChunks = [];


    setStatus(
        "Analyzing audio..."
    );


    // Send to Flask

    sendAudioBlob(
        wavBlob,
        "recording.wav"
    );

}


// ======================================================
// MERGE AUDIO CHUNKS
// ======================================================

function mergeBuffers(chunks) {

    let totalLength = 0;

    chunks.forEach(chunk => {

        totalLength += chunk.length;

    });


    const result =
        new Float32Array(totalLength);


    let offset = 0;


    chunks.forEach(chunk => {

        result.set(
            chunk,
            offset
        );

        offset += chunk.length;

    });


    return result;

}


// ======================================================
// DOWNSAMPLE AUDIO
// ======================================================

function downsampleBuffer(
    buffer,
    inputRate,
    outputRate
) {

    if (inputRate === outputRate) {

        return buffer;

    }


    const ratio =
        inputRate / outputRate;


    const newLength =
        Math.round(
            buffer.length / ratio
        );


    const result =
        new Float32Array(newLength);


    let offsetResult = 0;

    let offsetBuffer = 0;


    while (
        offsetResult < result.length
    ) {

        const nextOffsetBuffer =
            Math.round(
                (offsetResult + 1)
                * ratio
            );


        let accumulator = 0;

        let count = 0;


        for (
            let i = offsetBuffer;
            i < nextOffsetBuffer &&
            i < buffer.length;
            i++
        ) {

            accumulator += buffer[i];

            count++;

        }


        result[offsetResult] =
            count > 0
                ? accumulator / count
                : 0;


        offsetResult++;

        offsetBuffer =
            nextOffsetBuffer;

    }


    return result;

}


// ======================================================
// ENCODE WAV
// ======================================================

function encodeWAV(
    samples,
    sampleRate
) {

    const buffer =
        new ArrayBuffer(
            44 + samples.length * 2
        );


    const view =
        new DataView(buffer);


    // RIFF

    writeString(
        view,
        0,
        "RIFF"
    );


    view.setUint32(
        4,
        36 + samples.length * 2,
        true
    );


    // WAVE

    writeString(
        view,
        8,
        "WAVE"
    );


    // fmt

    writeString(
        view,
        12,
        "fmt "
    );


    view.setUint32(
        16,
        16,
        true
    );


    // PCM format

    view.setUint16(
        20,
        1,
        true
    );


    // Mono

    view.setUint16(
        22,
        1,
        true
    );


    // Sample rate

    view.setUint32(
        24,
        sampleRate,
        true
    );


    // Byte rate

    view.setUint32(
        28,
        sampleRate * 2,
        true
    );


    // Block align

    view.setUint16(
        32,
        2,
        true
    );


    // Bits per sample

    view.setUint16(
        34,
        16,
        true
    );


    // data

    writeString(
        view,
        36,
        "data"
    );


    view.setUint32(
        40,
        samples.length * 2,
        true
    );


    // Convert float → 16-bit PCM

    let offset = 44;


    for (
        let i = 0;
        i < samples.length;
        i++,
        offset += 2
    ) {

        const sample =
            Math.max(
                -1,
                Math.min(
                    1,
                    samples[i]
                )
            );


        const value =
            sample < 0
                ? sample * 0x8000
                : sample * 0x7fff;


        view.setInt16(
            offset,
            value,
            true
        );

    }


    return new Blob(
        [view],
        {
            type: "audio/wav"
        }
    );

}


// ======================================================
// WRITE STRING
// ======================================================

function writeString(
    view,
    offset,
    string
) {

    for (
        let i = 0;
        i < string.length;
        i++
    ) {

        view.setUint8(
            offset + i,
            string.charCodeAt(i)
        );

    }

}


// ======================================================
// SEND AUDIO TO FLASK
// ======================================================

async function sendAudioBlob(
    blob,
    filename = "audio.wav"
) {

    const formData =
        new FormData();


    // IMPORTANT:
    // app.py expects "file"

    formData.append(
        "file",
        blob,
        filename
    );


    try {

        setStatus(
            "Uploading audio and analyzing..."
        );


        const response =
            await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        // Check HTTP status

        if (!response.ok) {

            let errorMessage =
                `Server returned ${response.status}`;

            try {

                const errorData =
                    await response.json();

                if (errorData.error) {

                    errorMessage =
                        errorData.error;

                }

            } catch (_) {

                // Ignore JSON parsing error

            }


            throw new Error(
                errorMessage
            );

        }


        const data =
            await response.json();


        console.log(
            "Prediction response:",
            data
        );


        // Backend returned an error

        if (!data.success) {

            setStatus(
                `Error: ${
                    data.error ||
                    "Prediction failed."
                }`
            );

            setBoardIdle();

            return;

        }


        // Successful prediction

        setStatus(
            "Done."
        );


        updateBoard(
            data.label,
            data.confidence,
            data.breakdown
        );


    } catch (error) {

        console.error(
            "Prediction request failed:",
            error
        );


        setStatus(
            `Error: ${error.message}`
        );


        setBoardIdle();

    }

}


// ======================================================
// UPDATE RESULT BOARD
// ======================================================

function updateBoard(
    label,
    confidence,
    breakdown
) {

    board.classList.remove(
        "idle",
        "real",
        "fake"
    );


    const normalizedLabel =
        String(label)
            .toUpperCase();


    if (
        normalizedLabel === "REAL"
    ) {

        board.classList.add(
            "real"
        );


        boardLabel.textContent =
            "🟢 REAL";

    } else {

        board.classList.add(
            "fake"
        );


        boardLabel.textContent =
            "🔴 FAKE";

    }


    boardConfidence.textContent =
        `Confidence: ${confidence}%`;


    if (
        breakdown &&
        typeof breakdown === "object"
    ) {

        boardBreakdown.textContent =
            Object.entries(breakdown)
                .map(
                    ([key, value]) =>
                        `${key}: ${value}%`
                )
                .join("   ");

    } else {

        boardBreakdown.textContent =
            "";

    }

}


// ======================================================
// RESET RESULT BOARD
// ======================================================

function setBoardIdle() {

    board.classList.remove(
        "real",
        "fake"
    );


    board.classList.add(
        "idle"
    );


    boardLabel.textContent =
        "—";


    boardConfidence.textContent =
        "";


    boardBreakdown.textContent =
        "";

}


// ======================================================
// STATUS MESSAGE
// ======================================================

function setStatus(message) {

    statusEl.textContent =
        message;

}