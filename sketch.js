// C:\Users\Tal\Desktop\Hadar\ML5\reps_counter> py -m http.server

// p5 video size
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;
const modelDir = "model2";
const TRAINING_DATA_PATH = `${modelDir}/training_data.json`;

let video;
let poseNet;
let pose;
let skeleton;

let brain;
let lastPoseLabel;
let poseLabel;
let poseConfidence;
let lastSpeech = 0; // the frame count when the last speech was
let lastLog = "";

const SEPARATOR = "_";
const EXERCISES = ["pushups", "pullups"];
const DIRECTIONS = ["up", "down"];
let repsCounter = {}; // {pushups: 0, pullups: 0}
let samplesCounter = {}; // {pushups_up: 0, pushups_down: 0, pullups_up: 0, pullups_down: 0}
let resultsCounter = {}; // {  pushups: {up: 0, down: 0},  pullups: {up: 0, down: 0}  }

let autoCollecting = false;
let speech = true;


function setup() {
    // createCanvas(640, 480); // the size of the video
    createCanvas(VIDEO_WIDTH * 2, VIDEO_HEIGHT);

    video = createCapture(VIDEO);
    video.hide();
    poseNet = ml5.poseNet(video, posenetLoaded);
    poseNet.on('pose', gotPoses);

    let options = {
        inputs: 34, // posenet returns 17 points, each one has x, y (17*2 = 34)
        outputs: 4, // 2 exercises, each one has up and down positions
        task: 'classification',
        debug: true
    }
    brain = ml5.neuralNetwork(options);
    // loadBrain();


    for (let exercise of EXERCISES) {
        repsCounter[exercise] = 0;
        resultsCounter[exercise] = {};
        for (let direction of DIRECTIONS) {
            let label = exercise + SEPARATOR + direction;
            samplesCounter[label] = 0;
            resultsCounter[exercise][direction] = 0;


            createDiv(); // just for spacing
            let labelButton = createButton(label);
            labelButton.addClass("labelsButtons");

            let span = createSpan("0");
            span.id(label);
            span.addClass("countersElements");

            labelButton.mousePressed(function () {
                // let buttonLabel = this.elt.textContent;
                if (autoCollecting) {
                    const startDelay = 3000;

                    showLog(`start in ${startDelay/1000} seconds`);
                    setTimeout(function () {
                        showLog("collecting data...");
                        let interval = setInterval(function () {
                            addData(label)
                        }, 100);
                        setTimeout(function () {
                            showLog("finished collecting data");
                            clearInterval(interval);
                        }, 6000);
                    }, startDelay);

                } else {
                    addData(label);
                }
            });



        }
        createP(); // just for spacing
    }


    let trainButton = createButton("train");
    trainButton.addClass("labelsButtons");
    trainButton.mousePressed(function () {
        trainModel();
    });


    createDiv(); // just for spacing
    let saveDataButton = createButton("save data");
    saveDataButton.addClass("labelsButtons");
    saveDataButton.mousePressed(function () {
        brain.saveData();
    });

    let saveModelButton = createButton("save model");
    saveModelButton.addClass("labelsButtons");
    saveModelButton.mousePressed(function () {
        brain.save();
    });


    createDiv(); // just for spacing
    let loadDataButton = createButton("load data");
    loadDataButton.addClass("labelsButtons");
    loadDataButton.mousePressed(function () {
        // LOAD TRAINING DATA
        brain.loadData(TRAINING_DATA_PATH, dataLoaded);
        updateLabelsCounter();
    });

    let loadModelButton = createButton("load model");
    loadModelButton.addClass("labelsButtons");
    loadModelButton.mousePressed(function () {
        loadBrain();
    });


}

function showLog(log) {
    console.log(log);
    lastLog = log;
}

function speak(text) {
    // speak a text out loud
    let msg = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(msg);
}

function addData(targetLabel) {
    if (pose) {
        samplesCounter[targetLabel]++;
        document.getElementById(targetLabel).textContent = samplesCounter[targetLabel];

        let inputs = [];
        for (let i = 0; i < pose.keypoints.length; i++) {
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            inputs.push(x);
            inputs.push(y);
        }
        let target = [targetLabel];
        brain.addData(inputs, target);
    }
}

function updateLabelsCounter() {
    // let data = loadJSON(TRAINING_DATA_PATH);
    // console.log(data);
}

function dataLoaded() {
    showLog("data loaded");
}

function brainLoaded() {
    showLog('brain loaded');
    classifyPose();
}

function loadBrain() {
    // LOAD PRETRAINED MODEL
    const modelInfo = {
        model: `${modelDir}/model.json`,
        metadata: `${modelDir}/model_meta.json`,
        weights: `${modelDir}/model.weights.bin`
    };
    brain.load(modelInfo, brainLoaded);
}

function classifyPose() {
    if (pose) {
        let inputs = [];
        for (let i = 0; i < pose.keypoints.length; i++) {
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            inputs.push(x);
            inputs.push(y);
        }
        brain.classify(inputs, gotResult);
    } else {
        setTimeout(classifyPose, 100);
    }
}



function resetResultsCounter() {
    console.log("resetResultsCounter");
    console.log(JSON.stringify(resultsCounter));

    for (let ex in resultsCounter) {
        for (let dir in resultsCounter[ex]) {
            resultsCounter[ex][dir] = 0;
        }
    }
}

let minNumForRep = 4;
let maxNumWithoutRep = 30; // after that number reset all the counters

function gotResult(error, results) {
    if (results) {
        if (results[0].confidence > 0.75) {
            lastPoseLabel = poseLabel;
            poseLabel = results[0].label;
            poseConfidence = results[0].confidence.toPrecision(3); // keep only 3 digits after decimal point

            let [exercise, direction] = poseLabel.split(SEPARATOR);
            resultsCounter[exercise][direction]++;

            let anyValidRep = false;
            for (let [exercise, repsAtEachDir] of Object.entries(resultsCounter)) {
                if (min(Object.values(repsAtEachDir)) > minNumForRep) {
                    repsCounter[exercise]++;
                    if (speech) {
                        speak(repsCounter[exercise]);
                    }
                    anyValidRep = true;
                    resetResultsCounter();
                    break;
                }
            }
            if (!anyValidRep) {
                for (let [exercise, repsAtEachDir] of Object.entries(resultsCounter)) {
                    if (max(Object.values(repsAtEachDir)) > maxNumWithoutRep) {
                        resetResultsCounter();
                        break;
                    }
                }
            }


        }
    }
    classifyPose();
}


function trainModel() {
    brain.normalizeData();
    brain.train({
        epochs: 50
    }, finished);
}

function finished() {
    showLog('model trained');
    // brain.save();
    classifyPose();
}


function gotPoses(poses) {
    if (poses.length > 0) {
        pose = poses[0].pose;
        skeleton = poses[0].skeleton;
    }
}


function posenetLoaded() {
    showLog('POSENET is ready');
}

function draw() {
    background(0, 0, 128);
    push();
    translate(video.width, 0);
    scale(-1, 1);
    image(video, 0, 0, video.width, video.height);

    if (pose) {
        for (let i = 0; i < skeleton.length; i++) {
            let a = skeleton[i][0];
            let b = skeleton[i][1];
            strokeWeight(2);
            stroke(0);

            line(a.position.x, a.position.y, b.position.x, b.position.y);
        }
        for (let i = 0; i < pose.keypoints.length; i++) {
            let x = pose.keypoints[i].position.x;
            let y = pose.keypoints[i].position.y;
            fill(0);
            stroke(255);
            ellipse(x, y, 16, 16);
        }
    }
    pop();

    fill(255);
    noStroke();
    textAlign(LEFT);
    textSize(72);

    const rowHeight = height / (EXERCISES.length + 4);
    let i = 1;
    for (let [exercise, reps] of Object.entries(repsCounter)) {
        text(exercise + ":", VIDEO_WIDTH + 80, rowHeight * i);
        text(reps, VIDEO_WIDTH + 480, rowHeight * i);
        i++;
    }

    textAlign(CENTER, CENTER);
    textSize(48);
    if (poseLabel) {
        text(poseLabel, VIDEO_WIDTH / 2 + width / 2, height - 2 * rowHeight);
        textSize(32);
        text(`confidence: ${poseConfidence}`, VIDEO_WIDTH / 2 + width / 2, height - rowHeight);
    } else {
        text(lastLog, VIDEO_WIDTH / 2 + width / 2, height - 2 * rowHeight);
    }

    // if (speech) {
    //     if (poseLabel != lastPoseLabel) {
    //         if (frameCount - lastSpeech > 10) {
    //             speak(poseLabel.replace(SEPARATOR, " "));
    //             lastSpeech = frameCount;
    //         }
    //     }
    // }
}