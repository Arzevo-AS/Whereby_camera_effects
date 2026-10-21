// @ts-nocheck
// The canvas engine uses 2d canvas and filtered drawing to
// to achieve desired effects. It runs the segmentation model
// to separate person from background

import {
    loadSegmentationModel,
    SEGMENTATIONMODEL_TYPE_BACKGROUND_PERSON,
    SEGMENTATIONMODEL_TYPE_PERSON,
} from "../segmentationModel";

import { createCanvas } from "../../shared";
import { DrawingUtils, FaceLandmarker, FaceLandmarkerResult, FilesetResolver, NormalizedLandmark } from "@mediapipe/tasks-vision";

const baseCanvasParams = {
    maskOperation: "copy",
    maskFilter: "blur(4px)",
    personOperation: "source-in",
    personFilter: "none",
    personFillColor: "none",
    backgroundOperation: "destination-over",
    backgroundFilter: "none",
    backgroundFillColor: "none",
};

function getBlurCanvasParams(amount) {
    switch (amount) {
        case "slight": {
            return { ...baseCanvasParams, backgroundFilter: "blur(4px)", crop: 4 };
        }
        case "heavy": {
            return { ...baseCanvasParams, maskFilter: "blur(8px)", backgroundFilter: "blur(16px)", crop: 16 };
        }
    }
    // default
    return { ...baseCanvasParams, maskFilter: "blur(8px)", backgroundFilter: "blur(8px)", crop: 8 };
}

function getAnonymizationCanvasParams(type, amount, greyscale, color, applyBackground) {
    const params = { ...baseCanvasParams, backgroundFilter: "none" };
    switch (type) {
        case "pixelation": {
            ///TODO: Implement pixelation with WebGL shader for better performance and quality, currently it is done with canvas blur which is not ideal
            const pixelationPerson = amount === "slight" ? 8 : amount === "heavy" ? 20 : 12;
            params.personFilter = `blur(${pixelationPerson}px)`;
            params.maskFilter = "none"; // to avoid blurring the edges of the pixelation
            if (applyBackground) {
                params.backgroundFilter = `blur(${pixelationPerson}px)`;
            }
            break;
        }
        case "blur": {
            const blurAmount = amount === "slight" ? 4 : amount === "heavy" ? 16 : 8;
            params.personFilter = `blur(${blurAmount}px)`;
            params.maskFilter = "none"; // to avoid blurring the edges of the pixelation
            if (applyBackground) {
                params.backgroundFilter = `blur(${blurAmount}px)`;
            } else {
                params.maskFilter = "blur(8px)"; // Blurring the egde around the person and background
            }
            if (greyscale) {
                params.personFilter += " grayscale(100%)";
                if (applyBackground) {
                    params.backgroundFilter += " grayscale(100%)";
                }
            }
            break;
        }
        case "silhouette": {
            params.personFilter = "brightness(0%)";
            params.maskFilter = "blur(8px)"; // to avoid blurring the edges of the pixelation
            params.backgroundFilter = "none";
            if (applyBackground) {
                params.backgroundFillColor = color || "#ffffff";
            }
            break;
        }
        /// In color we fill in the whole image with the chosen. This filter is for when an avatar is used over the person.
        /// As it is not the same effect could be achieved by turning off the camera.
        case "color": {
            params.personFillColor = color || "#ffffff";
            params.maskFilter = "none";
            params.backgroundFilter = "none";
            if (applyBackground) {
                params.backgroundFillColor = color || "#ffffff";
            }
            break;
        }
    }

    return params;
}

const LAYER_BLUR_PX = { slight: 4, normal: 8, heavy: 16 };

function getLayerBlurFilter(effect) {
    const filter = `blur(${LAYER_BLUR_PX[effect.amount] ?? LAYER_BLUR_PX.normal}px)`;
    return effect.greyscale ? `${filter} grayscale(100%)` : filter;
}

// Person and background each get their own effect. Without either the frame is passed through
// as is, skipping segmentation.
function getLayerCanvasParams({ person, background }) {
    if (!person && !background) {
        return { ...baseCanvasParams, passthrough: true };
    }
    // Softens the edge between the layers, unless both are blurred anyway.
    const params = { ...baseCanvasParams, maskFilter: "blur(8px)" };
    if (person?.type === "blur") {
        params.personFilter = getLayerBlurFilter(person);
    } else if (person?.type === "fill") {
        params.personFillColor = person.color;
    }
    if (background?.type === "blur") {
        params.backgroundFilter = getLayerBlurFilter(background);
        // A blurred background darkens towards the frame edges, so it's cropped away.
        params.crop = LAYER_BLUR_PX[background.amount] ?? LAYER_BLUR_PX.normal;
    } else if (background?.type === "fill") {
        params.backgroundFillColor = background.color;
    }
    if (person?.type === "blur" && background?.type === "blur") {
        params.maskFilter = "none";
    }
    return params;
}

function getCanvasParams(params) {
    if (params.layers) return getLayerCanvasParams(params.layers);
    if (params.backgroundBlur) return getBlurCanvasParams(params.backgroundBlur.amount);
    if (params.anonymization) return getAnonymizationCanvasParams(params.anonymization.type, 
        params.anonymization.amount, params.anonymization.greyscale, params.anonymization.color, params.anonymization.applyBackground);
    return baseCanvasParams;
}

const DEFAULT_AVATAR_COLOR = "#03fa07";

const baseAvatarParams = {
    enabled: false,
    color: DEFAULT_AVATAR_COLOR,
};

function getAvatarParams(params) {
    const avatar = params.layers ? params.layers.avatar : params.avatar;
    return getAvatarCanvasParams(avatar?.type, avatar?.color);
}

function getAvatarCanvasParams(type, color) {
    const params = { ...baseAvatarParams, color: color || DEFAULT_AVATAR_COLOR };
    switch (type) {
        case "wireframe": {
            params.enabled = true;
            break;
        }
        case "2d": {
            params.enabled = true;
            break;
        }
        case "3d": {
            params.enabled = true;
            // 3D avatar rendering is not supported in canvas engine, it requires WebGL. This is a placeholder for future implementation.
            break;
        }
    }
    return params;
}

// Pinned to the installed @mediapipe/tasks-vision, so the wasm always matches its JS API.
const DEFAULT_FACE_LANDMARKER_WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const DEFAULT_FACE_LANDMARKER_MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const WIREFRAME_FEATURES = [
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYE,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_EYEBROW,
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYE,
    FaceLandmarker.FACE_LANDMARKS_LEFT_EYEBROW,
    FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
    FaceLandmarker.FACE_LANDMARKS_LIPS,
    FaceLandmarker.FACE_LANDMARKS_RIGHT_IRIS,
    FaceLandmarker.FACE_LANDMARKS_LEFT_IRIS,
];

// Shared by every engine, as an engine is recreated whenever the video size changes. Never
// resolves after a failed load, so a broken landmarker is not retried every frame.
let faceLandmarkerPromise = null;

function loadFaceLandmarker(setup) {
    return (async () => {
        const vision = await FilesetResolver.forVisionTasks(setup.faceLandmarkerWasmUrl || DEFAULT_FACE_LANDMARKER_WASM_URL);
        ///TODO: Look into using GPU instead
        return await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: setup.faceLandmarkerModelUrl || DEFAULT_FACE_LANDMARKER_MODEL_URL,
                delegate: "CPU",
            },
            numFaces: 1,
            minDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
            minSuppressionThreshold: 0.5,
            runningMode: "VIDEO",
        });
    })().catch((error) => {
        console.error("Failed to load the face landmarker, avatars are not drawn", error);
        return new Promise(() => undefined);
    });
}

export async function createCanvasEngine(videoWidth, videoHeight, setup, effectCanvas, params) {
    // tflite, model
    const {
        tflite,
        model: segmentationModel,
        inputHeight,
        inputWidth,
        inputMemoryOffset,
        outputMemoryOffset,
        segmentationPixelCount,
    } = await loadSegmentationModel(setup.segmentationModelId);

    // Loaded on the first frame that needs an avatar, so effects without one never pay for it.
    let faceLandmarker = null;
    let isFaceLandmarkerRequested = false;
    const ensureFaceLandmarker = () => {
        if (isFaceLandmarkerRequested) return;
        isFaceLandmarkerRequested = true;
        if (!faceLandmarkerPromise) {
            faceLandmarkerPromise = loadFaceLandmarker(setup);
        }
        faceLandmarkerPromise.then((landmarker) => (faceLandmarker = landmarker));
    };

    const drawWireframe = (drawingUtils: DrawingUtils, landmarks: NormalizedLandmark[], color: string) => {
        // Hex alpha suffix, so the mesh stays fainter than the features drawn over it.
        drawingUtils.drawConnectors(landmarks, FaceLandmarker.FACE_LANDMARKS_TESSELATION, { color: `${color}70`, lineWidth: 1 });
        for (const connections of WIREFRAME_FEATURES) {
            drawingUtils.drawConnectors(landmarks, connections, { color });
        }
    };

    let canvasParams = getCanvasParams(params);
    let avatarParams = getAvatarParams(params);

    // canvas for rendering images and running segmentation model
    const segmentationCanvas = createCanvas(inputWidth, inputHeight);
    const segmentationCtx = segmentationCanvas.getContext("2d", {
        willReadFrequently: true,
    });

    // image holding result of segmentation model
    const segmentationMask = new ImageData(inputWidth, inputHeight);

    let currentBackgroundFrame = null;
    let backgroundDimensions = { x: 0, y: 0, width: 0, height: 0 };

    // we need to know the video aspect ratio for background and cropping
    const videoAspectRatio = videoWidth / videoHeight;

    // calculates/updates a crop respecting the aspect ratio
    let cropX = 0;
    let cropY = 0;
    const updateCrop = () => {
        const crop = canvasParams.crop || 0;
        if (crop) {
            cropX = videoAspectRatio > 1 ? Math.round(crop * videoAspectRatio) : crop;
            cropY = videoAspectRatio < 1 ? Math.round(crop / videoAspectRatio) : crop;
        } else {
            cropX = 0;
            cropY = 0;
        }
    };
    updateCrop();

    const effectCtx = effectCanvas.getContext("2d");
    const utils = new DrawingUtils(effectCtx);

    const drawAvatar = (frame) => {
        if (!avatarParams.enabled) return;
        if (!faceLandmarker) {
            ensureFaceLandmarker();
            return;
        }
        //TODO: Is this the correct frame to run detection on?
        const result: FaceLandmarkerResult = faceLandmarker.detectForVideo(frame, performance.now());
        if (result.faceLandmarks.length > 0) {
            effectCtx.globalCompositeOperation = "source-over";
            effectCtx.filter = "none";
            for (const landmarks of result.faceLandmarks) {
                drawWireframe(utils, landmarks, avatarParams.color);
            }
        }
    };

    return {
        effectCtx,
        updateBackgroundFrame(frame, dimensions) {
            currentBackgroundFrame = frame;
            if (dimensions) backgroundDimensions = dimensions;
        },
        updateParams(updatedParams) {
            canvasParams = getCanvasParams(updatedParams);
            avatarParams = getAvatarParams(updatedParams);
            updateCrop();
        },
        processFrame(frame) {
            if (canvasParams.passthrough) {
                effectCtx.globalCompositeOperation = "copy";
                effectCtx.filter = "none";
                effectCtx.drawImage(frame, 0, 0);
                drawAvatar(frame);
                return;
            }

            // copy and resize video to segmentation canvas
            segmentationCtx.drawImage(frame, 0, 0, videoWidth, videoHeight, 0, 0, inputWidth, inputHeight);

            // fill model with input
            const imageData = segmentationCtx.getImageData(0, 0, inputWidth, inputHeight);
            for (let i = 0; i < segmentationPixelCount; i++) {
                tflite.HEAPF32[inputMemoryOffset + i * 3] = imageData.data[i * 4] / 255;
                tflite.HEAPF32[inputMemoryOffset + i * 3 + 1] = imageData.data[i * 4 + 1] / 255;
                tflite.HEAPF32[inputMemoryOffset + i * 3 + 2] = imageData.data[i * 4 + 2] / 255;
            }

            // run model
            tflite._runInference();

            // create segmentation mask image from model result
            for (let i = 0; i < segmentationPixelCount; i++) {
                if (segmentationModel.type === SEGMENTATIONMODEL_TYPE_BACKGROUND_PERSON) {
                    const background = tflite.HEAPF32[outputMemoryOffset + i * 2];
                    const person = tflite.HEAPF32[outputMemoryOffset + i * 2 + 1];
                    const shift = Math.max(background, person);
                    const backgroundExp = Math.exp(background - shift);
                    const personExp = Math.exp(person - shift);
                    segmentationMask.data[i * 4 + 3] = (255 * personExp) / (backgroundExp + personExp);
                } else if (segmentationModel.type === SEGMENTATIONMODEL_TYPE_PERSON) {
                    const person = tflite.HEAPF32[outputMemoryOffset + i];
                    segmentationMask.data[i * 4 + 3] = 255 * person;
                }
            }

            // render result back to segmentation canvas
            segmentationCtx.putImageData(segmentationMask, 0, 0);

            // draw mask with blur on effect canvas
            effectCtx.globalCompositeOperation = canvasParams.maskOperation;
            effectCtx.filter = canvasParams.maskFilter; // FIXME Does not work on Safari
            effectCtx.drawImage(segmentationCanvas, 0, 0, inputWidth, inputHeight, 0, 0, videoWidth, videoHeight);

            // draw person with effects
            effectCtx.globalCompositeOperation = canvasParams.personOperation;
            effectCtx.filter = canvasParams.personFilter;
            if (canvasParams.personFillColor !== "none") {
                effectCtx.fillStyle = canvasParams.personFillColor;
                effectCtx.fillRect(0, 0, videoWidth, videoHeight);
            } else {
                effectCtx.drawImage(frame, 0, 0);
            }

            drawAvatar(frame);

            // draw background with effects
            effectCtx.globalCompositeOperation = canvasParams.backgroundOperation;
            effectCtx.filter = canvasParams.backgroundFilter; // Filters does not work on Safari
            if (currentBackgroundFrame) {
                effectCtx.drawImage(
                    currentBackgroundFrame,
                    backgroundDimensions.x,
                    backgroundDimensions.y,
                    backgroundDimensions.width,
                    backgroundDimensions.height,
                    0,
                    0,
                    videoWidth,
                    videoHeight,
                    0,
                );
            } else {
                if (canvasParams.backgroundFillColor !== "none") {
                    effectCtx.fillStyle = canvasParams.backgroundFillColor;
                    effectCtx.fillRect(0, 0, videoWidth, videoHeight);
                }
                effectCtx.drawImage(frame, 0, 0);
            }

            // crop and rescale image
            if (cropX || cropY) {
                effectCtx.globalCompositeOperation = "source-over";
                effectCtx.filter = "none";

                effectCtx.drawImage(
                    effectCanvas,
                    cropX,
                    cropY,
                    videoWidth - cropX * 2,
                    videoHeight - cropY * 2,
                    0,
                    0,
                    videoWidth,
                    videoHeight,
                );
            }
        },
        dispose() {},
    };
}
