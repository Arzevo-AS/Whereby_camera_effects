// @ts-nocheck
import { loadSegmentationModel } from "../../segmentationModel";

import { compileShader, createTexture, glsl } from "./webglHelper";
import { buildBackgroundBlurStage } from "./backgroundBlurStage";
import { buildBackgroundImageStage } from "./backgroundImageStage";
import { buildAnonymizationStage } from "./anonymizationStage";
import { buildImproveMaskStage } from "./improveMaskStage";
import { buildTFLiteOutputToMaskStage } from "./tfliteOutputToMaskStage";
import { buildTFLiteInputStage } from "./tfliteInputStage";

const MODE_NONE = 0;
const MODE_BLUR = 1;
const MODE_SILHOUETTE = 2;
const MODE_COLOR = 3;

function getBlurEngineParams(params) {
    switch (params.amount) {
        case "slight": {
            return { kernel: "og5", numPasses: 1, backgroundResampleScale: 1, ...params };
        }
        case "heavy": {
            return { kernel: "og10", numPasses: 1, backgroundResampleScale: 0.5, ...params };
        }
        default: {
            return { kernel: "og5", numPasses: 1, backgroundResampleScale: 0.5, ...params };
        }
    }
}

function getAnonymizationEngineParams(params) {
    const base = {
        personMode: MODE_NONE,
        personBlurRadius: 0,
        blurKernel: "og8",
        personGreyscale: false,
        personColor: params.color || "#ffffff",
        backgroundMode: MODE_NONE,
        backgroundBlurRadius: 0,
        backgroundGreyscale: false,
        backgroundColor: params.color || "#ffffff",
        maskFeatherPx: 0,
    };

    switch (params.type) {
        case "pixelation": {
            const pixelationAmount = params.amount === "slight" ? 0.8 : params.amount === "heavy" ? 1.8 : 1.2;
            return {
                ...base,
                blurKernel: "og4",
                personMode: MODE_BLUR,
                personBlurRadius: pixelationAmount,
                backgroundMode: params.applyBackground ? MODE_BLUR : MODE_NONE,
                backgroundBlurRadius: params.applyBackground ? pixelationAmount : 0,
            };
        }
        case "blur": {
            const blurAmount = params.amount === "slight" ? 0.9 : params.amount === "heavy" ? 2.8 : 1.7;
            const blurKernel = params.amount === "slight" ? "og6" : params.amount === "heavy" ? "og12" : "og9";
            return {
                ...base,
                blurKernel,
                personMode: MODE_BLUR,
                personBlurRadius: blurAmount,
                personGreyscale: !!params.greyscale,
                backgroundMode: params.applyBackground ? MODE_BLUR : MODE_NONE,
                backgroundBlurRadius: params.applyBackground ? blurAmount : 0,
                backgroundGreyscale: params.applyBackground && !!params.greyscale,
                // Mirror canvas behavior where blur-person mode softens mask edge.
                maskFeatherPx: params.applyBackground ? 0 : 8,
            };
        }
        case "silhouette": {
            return {
                ...base,
                personMode: MODE_SILHOUETTE,
                backgroundMode: params.applyBackground ? MODE_COLOR : MODE_NONE,
                backgroundColor: params.color || "#ffffff",
                // Mirror canvas mask blur for silhouette mode.
                maskFeatherPx: 8,
            };
        }
        case "color": {
            return {
                ...base,
                personMode: MODE_COLOR,
                personColor: params.color || "#ffffff",
                backgroundMode: params.applyBackground ? MODE_COLOR : MODE_NONE,
                backgroundColor: params.color || "#ffffff",
            };
        }
        default:
            return base;
    }
}

function getEngineParams(params) {
    const baseEngineParams = {
        coverage: params.coverage,
    };
    if (params.backgroundBlur) {
        return { ...baseEngineParams, ...getBlurEngineParams(params.backgroundBlur) };
    }
    if (params.anonymization) return { ...baseEngineParams, ...getAnonymizationEngineParams(params.anonymization) };
    return baseEngineParams;
}

// The webgl engine uses WebGL2 shaders
// both for loading segmentation model with data and achieve desired effects
export async function createWebGLEngine(videoWidth, videoHeight, setup, effectCanvas, params, initialBackgroundFrame) {
    // tflite, model
    const { tflite, inputHeight, inputWidth, inputMemoryOffset, outputMemoryOffset, segmentationPixelCount } =
        await loadSegmentationModel(setup.segmentationModelId);

    let engineParams = getEngineParams(params);

    const vertexShaderSource = glsl`#version 300 es
        in vec2 a_position;
        in vec2 a_texCoord;
        out vec2 v_texCoord;
        void main() {
            gl_Position = vec4(a_position, 0.0, 1.0);
            v_texCoord = a_texCoord;
        }`;

    const gl = effectCanvas.getContext("webgl2", setup.webglContextSettings);
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);

    const vertexArray = gl.createVertexArray();
    gl.bindVertexArray(vertexArray);

    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]), gl.STATIC_DRAW);

    const texCoordBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, texCoordBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), gl.STATIC_DRAW);

    const inputFrameTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const maskTexture = createTexture(gl, gl.RGBA8, inputWidth, inputHeight);

    // TODO: play around with resolution
    const improvedMaskWidth = inputWidth;
    const improvedMaskHeight = inputHeight;
    const improvedMaskTexture = createTexture(
        gl,
        gl.RGBA8,
        improvedMaskWidth,
        improvedMaskHeight,
        gl.LINEAR,
        gl.LINEAR,
    );
    const tfliteInputStage = buildTFLiteInputStage(
        gl,
        vertexShader,
        positionBuffer,
        texCoordBuffer,
        tflite,
        inputWidth,
        inputHeight,
        segmentationPixelCount,
        inputMemoryOffset,
    );
    const tfliteOutputToMaskStage = buildTFLiteOutputToMaskStage(
        gl,
        vertexShader,
        positionBuffer,
        texCoordBuffer,
        maskTexture,
        tflite,
        inputWidth,
        inputHeight,
        outputMemoryOffset,
    );
    const improveMaskStage = buildImproveMaskStage(
        gl,
        vertexShader,
        positionBuffer,
        texCoordBuffer,
        maskTexture,
        improvedMaskTexture,
        improvedMaskWidth,
        improvedMaskHeight,
    );
    const backgroundBlurStage = buildBackgroundBlurStage(
        gl,
        vertexShader,
        positionBuffer,
        texCoordBuffer,
        improvedMaskTexture,
        videoWidth,
        videoHeight,
        engineParams,
    );
    const backgroundImageStage = buildBackgroundImageStage(
        gl,
        positionBuffer,
        texCoordBuffer,
        improvedMaskTexture,
        initialBackgroundFrame,
        videoWidth,
        videoHeight,
        engineParams,
    );
    const anonymizationStage = buildAnonymizationStage(
        gl,
        positionBuffer,
        texCoordBuffer,
        improvedMaskTexture,
        improvedMaskWidth,
        improvedMaskHeight,
        initialBackgroundFrame,
        videoWidth,
        videoHeight,
        engineParams,
    );

    return {
        effectCtx: gl,
        updateBackgroundFrame(frame, _, reInit) {
            backgroundImageStage.updateBackgroundImage(frame, reInit);
            anonymizationStage.updateBackgroundImage(frame, reInit);
        },
        updateParams(updatedParams) {
            params = updatedParams;
            engineParams = getEngineParams(params);
            backgroundBlurStage.updateParams(engineParams);
            backgroundImageStage.updateParams(engineParams);
            anonymizationStage.updateParams(engineParams);
        },
        processFrame(frame) {
            gl.clearColor(0, 0, 0, 0);
            gl.clear(gl.COLOR_BUFFER_BIT);
            gl.activeTexture(gl.TEXTURE0);
            gl.bindTexture(gl.TEXTURE_2D, inputFrameTexture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, frame);
            gl.bindVertexArray(vertexArray);
            tfliteInputStage.render();
            tflite._runInference();
            tfliteOutputToMaskStage.render();
            improveMaskStage.render();
            if (params.anonymization) {
                anonymizationStage.render();
                return;
            }
            // eslint-disable-next-line
            params.backgroundUrl ? backgroundImageStage.render() : backgroundBlurStage.render();
        },
        dispose() {
            anonymizationStage.cleanUp();
            backgroundImageStage.cleanUp();
            backgroundBlurStage.cleanUp();
            improveMaskStage.cleanUp();
            tfliteOutputToMaskStage.cleanUp();
            tfliteInputStage.cleanUp();

            gl.deleteTexture(improvedMaskTexture);
            gl.deleteTexture(maskTexture);
            gl.deleteTexture(inputFrameTexture);
            gl.deleteBuffer(texCoordBuffer);
            gl.deleteBuffer(positionBuffer);
            gl.deleteVertexArray(vertexArray);
            gl.deleteShader(vertexShader);
        },
    };
}
