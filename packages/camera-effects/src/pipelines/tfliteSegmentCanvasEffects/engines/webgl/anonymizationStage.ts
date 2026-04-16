// @ts-nocheck
import { compileShader, createPiplelineStageProgram, createTexture, glsl } from "./webglHelper";

// Same optimized gaussian kernel strategy as backgroundBlurStage.
function generateOptimizedGaussianKernel(sigma) {
    const g = (x) => Math.exp(-(x * x) / (2 * sigma * sigma));

    const numbers = [];
    let x = 0;
    let currentSum = 0;

    // eslint-disable-next-line no-constant-condition
    while (true) {
        const n = g(x);
        const nextSum = currentSum + (currentSum ? n * 2 : n);
        if (currentSum && n / nextSum < 0.01 && numbers.length % 2) break;
        currentSum = nextSum;
        numbers.push(n);
        x++;
    }

    const weights = numbers.map((n) => n / currentSum);

    const optimizedWeights = [weights[0]];
    const optimizedPositions = [0];

    if (weights.length >= 3 && weights.length % 2 === 1) {
        for (let i = 1; i < weights.length; i += 2) {
            const w1 = weights[i];
            const w2 = weights[i + 1];
            const w = w1 + w2;
            const p = w2 / (w1 + w2);
            optimizedWeights.push(w);
            optimizedPositions.push(i + p);
        }
    }

    return {
        weights: optimizedWeights,
        positions: optimizedPositions,
    };
}

function parseColorToRGB(color, fallback = [1, 1, 1]) {
    if (!color || typeof color !== "string") return fallback;

    const value = color.trim().toLowerCase();
    if (value.startsWith("#")) {
        const hex = value.slice(1);
        if (hex.length === 3 || hex.length === 4) {
            const r = parseInt(hex[0] + hex[0], 16);
            const g = parseInt(hex[1] + hex[1], 16);
            const b = parseInt(hex[2] + hex[2], 16);
            if ([r, g, b].every((n) => Number.isFinite(n))) return [r / 255, g / 255, b / 255];
        }
        if (hex.length === 6 || hex.length === 8) {
            const r = parseInt(hex.slice(0, 2), 16);
            const g = parseInt(hex.slice(2, 4), 16);
            const b = parseInt(hex.slice(4, 6), 16);
            if ([r, g, b].every((n) => Number.isFinite(n))) return [r / 255, g / 255, b / 255];
        }
    }

    const rgbMatch = value.match(/^rgba?\(([^)]+)\)$/);
    if (rgbMatch?.[1]) {
        const [r, g, b] = rgbMatch[1]
            .split(",")
            .slice(0, 3)
            .map((n) => parseFloat(n.trim()));
        if ([r, g, b].every((n) => Number.isFinite(n))) {
            return [Math.min(255, Math.max(0, r)) / 255, Math.min(255, Math.max(0, g)) / 255, Math.min(255, Math.max(0, b)) / 255];
        }
    }

    return fallback;
}

const MODE_NONE = 0;
const MODE_BLUR = 1;
const MODE_SILHOUETTE = 2;
const MODE_COLOR = 3;

// Applies person/background anonymization effects using the segmentation mask.
export function buildAnonymizationStage(
    gl,
    positionBuffer,
    texCoordBuffer,
    personMaskTexture,
    personMaskWidth,
    personMaskHeight,
    initialBackgroundImage,
    outputWidth,
    outputHeight,
    params,
) {
    const vertexShaderSource = glsl`#version 300 es
        uniform vec2 u_backgroundScale;
        uniform vec2 u_backgroundOffset;
        in vec2 a_position;
        in vec2 a_texCoord;
        out vec2 v_texCoord;
        out vec2 v_backgroundCoord;
        void main() {
            // Flipping Y for canvas
            gl_Position = vec4(a_position * vec2(1.0, -1.0), 0.0, 1.0);
            v_texCoord = a_texCoord;
            v_backgroundCoord = a_texCoord * u_backgroundScale + u_backgroundOffset;
        }`;

    const fragmentShaderSource = glsl`#version 300 es
        precision highp float;
        uniform sampler2D u_inputFrame;
        uniform sampler2D u_personMask;
        uniform sampler2D u_backgroundFrame;
        uniform vec2 u_coverage;
        uniform vec2 u_inputTexelSize;
        uniform vec2 u_maskTexelSize;
        uniform float u_weight[20];
        uniform float u_offset[20];
        uniform int u_ksize;
        uniform float u_maskFeatherPx;
        uniform float u_useBackgroundTexture;

        uniform int u_personMode;
        uniform float u_personBlurRadius;
        uniform float u_personGreyscale;
        uniform vec3 u_personColor;

        uniform int u_backgroundMode;
        uniform float u_backgroundBlurRadius;
        uniform float u_backgroundGreyscale;
        uniform vec3 u_backgroundColor;

        in vec2 v_texCoord;
        in vec2 v_backgroundCoord;
        out vec4 outColor;

        vec3 toGreyscale(vec3 color) {
            float luma = dot(color, vec3(0.299, 0.587, 0.114));
            return vec3(luma);
        }

        vec3 sampleBlurInput(vec2 uv, float radiusPx) {
            if (radiusPx <= 0.01) return texture(u_inputFrame, uv).rgb;

            vec3 color = texture(u_inputFrame, uv).rgb * u_weight[0];
            float totalWeight = u_weight[0];

            for (int i = 1; i < u_ksize; i++) {
                float offset = u_offset[i] * radiusPx;
                vec2 dx = vec2(offset * u_inputTexelSize.x, 0.0);
                vec2 dy = vec2(0.0, offset * u_inputTexelSize.y);
                float w = u_weight[i];
                // Slightly lower diagonal weight avoids over-blur while removing plus-shaped artifacts.
                float wd = w * 0.85;
                // Keep diagonal distance similar to axis-aligned radius.
                float diagScale = 0.70710678;
                vec2 dd = vec2(dx.x * diagScale, dy.y * diagScale);

                color += texture(u_inputFrame, uv + dx).rgb * w;
                color += texture(u_inputFrame, uv - dx).rgb * w;
                color += texture(u_inputFrame, uv + dy).rgb * w;
                color += texture(u_inputFrame, uv - dy).rgb * w;
                color += texture(u_inputFrame, uv + dd).rgb * wd;
                color += texture(u_inputFrame, uv - dd).rgb * wd;
                color += texture(u_inputFrame, uv + vec2(dd.x, -dd.y)).rgb * wd;
                color += texture(u_inputFrame, uv + vec2(-dd.x, dd.y)).rgb * wd;
                totalWeight += 4.0 * w + 4.0 * wd;
            }

            return color / max(totalWeight, 0.0001);
        }

        vec3 sampleBlurBackground(vec2 uv, float radiusPx) {
            if (radiusPx <= 0.01) return texture(u_backgroundFrame, uv).rgb;

            vec3 color = texture(u_backgroundFrame, uv).rgb * u_weight[0];
            float totalWeight = u_weight[0];

            for (int i = 1; i < u_ksize; i++) {
                float offset = u_offset[i] * radiusPx;
                vec2 dx = vec2(offset * u_inputTexelSize.x, 0.0);
                vec2 dy = vec2(0.0, offset * u_inputTexelSize.y);
                float w = u_weight[i];
                float wd = w * 0.85;
                float diagScale = 0.70710678;
                vec2 dd = vec2(dx.x * diagScale, dy.y * diagScale);

                color += texture(u_backgroundFrame, uv + dx).rgb * w;
                color += texture(u_backgroundFrame, uv - dx).rgb * w;
                color += texture(u_backgroundFrame, uv + dy).rgb * w;
                color += texture(u_backgroundFrame, uv - dy).rgb * w;
                color += texture(u_backgroundFrame, uv + dd).rgb * wd;
                color += texture(u_backgroundFrame, uv - dd).rgb * wd;
                color += texture(u_backgroundFrame, uv + vec2(dd.x, -dd.y)).rgb * wd;
                color += texture(u_backgroundFrame, uv + vec2(-dd.x, dd.y)).rgb * wd;
                totalWeight += 4.0 * w + 4.0 * wd;
            }

            return color / max(totalWeight, 0.0001);
        }

        float samplePersonMask(vec2 uv) {
            float personMask = texture(u_personMask, uv).a;
            if (u_maskFeatherPx > 0.01) {
                vec2 featherOffset = u_maskTexelSize * u_maskFeatherPx;
                float blurredMask = 0.0;
                blurredMask += texture(u_personMask, uv).a * 4.0;
                blurredMask += texture(u_personMask, uv + vec2( featherOffset.x, 0.0)).a * 2.0;
                blurredMask += texture(u_personMask, uv + vec2(-featherOffset.x, 0.0)).a * 2.0;
                blurredMask += texture(u_personMask, uv + vec2(0.0,  featherOffset.y)).a * 2.0;
                blurredMask += texture(u_personMask, uv + vec2(0.0, -featherOffset.y)).a * 2.0;
                blurredMask += texture(u_personMask, uv + vec2( featherOffset.x,  featherOffset.y)).a;
                blurredMask += texture(u_personMask, uv + vec2(-featherOffset.x,  featherOffset.y)).a;
                blurredMask += texture(u_personMask, uv + vec2( featherOffset.x, -featherOffset.y)).a;
                blurredMask += texture(u_personMask, uv + vec2(-featherOffset.x, -featherOffset.y)).a;
                personMask = blurredMask / 16.0;
            }
            return smoothstep(u_coverage.x, u_coverage.y, personMask);
        }

        void main() {
            vec3 inputColor = texture(u_inputFrame, v_texCoord).rgb;

            vec3 backgroundSourceColor = inputColor;
            if (u_useBackgroundTexture > 0.5) {
                backgroundSourceColor = texture(u_backgroundFrame, v_backgroundCoord).rgb;
            }

            vec3 personColor = inputColor;
            if (u_personMode == 1) {
                personColor = sampleBlurInput(v_texCoord, u_personBlurRadius);
            } else if (u_personMode == 2) {
                personColor = vec3(0.0);
            } else if (u_personMode == 3) {
                personColor = u_personColor;
            }
            if (u_personGreyscale > 0.5) {
                personColor = toGreyscale(personColor);
            }

            vec3 backgroundColor = backgroundSourceColor;
            if (u_backgroundMode == 1) {
                if (u_useBackgroundTexture > 0.5) {
                    backgroundColor = sampleBlurBackground(v_backgroundCoord, u_backgroundBlurRadius);
                } else {
                    backgroundColor = sampleBlurInput(v_texCoord, u_backgroundBlurRadius);
                }
            } else if (u_backgroundMode == 3) {
                backgroundColor = u_backgroundColor;
            }
            if (u_backgroundGreyscale > 0.5) {
                backgroundColor = toGreyscale(backgroundColor);
            }

            float personMask = samplePersonMask(v_texCoord);
            outColor = vec4(mix(backgroundColor, personColor, personMask), 1.0);
        }`;

    const outputRatio = outputWidth / outputHeight;

    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
    const program = createPiplelineStageProgram(gl, vertexShader, fragmentShader, positionBuffer, texCoordBuffer);

    const backgroundScaleLocation = gl.getUniformLocation(program, "u_backgroundScale");
    const backgroundOffsetLocation = gl.getUniformLocation(program, "u_backgroundOffset");
    const inputFrameLocation = gl.getUniformLocation(program, "u_inputFrame");
    const personMaskLocation = gl.getUniformLocation(program, "u_personMask");
    const backgroundFrameLocation = gl.getUniformLocation(program, "u_backgroundFrame");
    const coverageLocation = gl.getUniformLocation(program, "u_coverage");
    const inputTexelSizeLocation = gl.getUniformLocation(program, "u_inputTexelSize");
    const maskTexelSizeLocation = gl.getUniformLocation(program, "u_maskTexelSize");
    const weightLocation = gl.getUniformLocation(program, "u_weight");
    const offsetLocation = gl.getUniformLocation(program, "u_offset");
    const kernelsizeLocation = gl.getUniformLocation(program, "u_ksize");
    const maskFeatherPxLocation = gl.getUniformLocation(program, "u_maskFeatherPx");
    const useBackgroundTextureLocation = gl.getUniformLocation(program, "u_useBackgroundTexture");
    const personModeLocation = gl.getUniformLocation(program, "u_personMode");
    const personBlurRadiusLocation = gl.getUniformLocation(program, "u_personBlurRadius");
    const personGreyscaleLocation = gl.getUniformLocation(program, "u_personGreyscale");
    const personColorLocation = gl.getUniformLocation(program, "u_personColor");
    const backgroundModeLocation = gl.getUniformLocation(program, "u_backgroundMode");
    const backgroundBlurRadiusLocation = gl.getUniformLocation(program, "u_backgroundBlurRadius");
    const backgroundGreyscaleLocation = gl.getUniformLocation(program, "u_backgroundGreyscale");
    const backgroundColorLocation = gl.getUniformLocation(program, "u_backgroundColor");

    gl.useProgram(program);
    gl.uniform1i(inputFrameLocation, 0);
    gl.uniform1i(personMaskLocation, 1);
    gl.uniform1i(backgroundFrameLocation, 2);
    gl.uniform2f(backgroundScaleLocation, 1, 1);
    gl.uniform2f(backgroundOffsetLocation, 0, 0);
    gl.uniform2f(inputTexelSizeLocation, 1 / outputWidth, 1 / outputHeight);
    gl.uniform2f(maskTexelSizeLocation, 1 / personMaskWidth, 1 / personMaskHeight);

    function updateKernel(kernel = "og10") {
        gl.useProgram(program);
        if (kernel.startsWith("og")) {
            const s = parseFloat(kernel.substring(2));
            const { weights, positions } = generateOptimizedGaussianKernel(s);
            gl.uniform1i(kernelsizeLocation, weights.length);
            gl.uniform1fv(weightLocation, weights);
            gl.uniform1fv(offsetLocation, positions);
        } else if (kernel === "7x7") {
            gl.uniform1i(kernelsizeLocation, 4);
            gl.uniform1fv(weightLocation, [0.3125, 0.234375, 0.09375, 0.015625]);
            gl.uniform1fv(offsetLocation, [0.0, 1.0, 2.0, 3.0]);
        } else if (kernel === "9x9") {
            gl.uniform1i(kernelsizeLocation, 5);
            gl.uniform1fv(weightLocation, [0.227027027, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162]);
            gl.uniform1fv(offsetLocation, [0.0, 1.0, 2.0, 3.0, 4.0]);
        } else {
            gl.uniform1i(kernelsizeLocation, 3);
            gl.uniform1fv(weightLocation, [0.3715, 0.25, 0.0625]);
            gl.uniform1fv(offsetLocation, [0.0, 1.0, 2.0]);
        }
    }

    let backgroundTexture = null;

    function initBackgroundImage(backgroundImage) {
        const sourceWidth = backgroundImage.naturalWidth || backgroundImage.videoWidth || backgroundImage.width;
        const sourceHeight = backgroundImage.naturalHeight || backgroundImage.videoHeight || backgroundImage.height;

        backgroundTexture = createTexture(gl, gl.RGBA8, sourceWidth, sourceHeight, gl.LINEAR, gl.LINEAR);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, sourceWidth, sourceHeight, gl.RGBA, gl.UNSIGNED_BYTE, backgroundImage);

        let xOffset = 0;
        let yOffset = 0;
        let backgroundWidth = sourceWidth;
        let backgroundHeight = sourceHeight;
        const backgroundRatio = backgroundWidth / backgroundHeight;
        if (backgroundRatio < outputRatio) {
            backgroundHeight = backgroundWidth / outputRatio;
            yOffset = (sourceHeight - backgroundHeight) / 2;
        } else {
            backgroundWidth = backgroundHeight * outputRatio;
            xOffset = (sourceWidth - backgroundWidth) / 2;
        }

        const xScale = backgroundWidth / sourceWidth;
        const yScale = backgroundHeight / sourceHeight;
        xOffset /= sourceWidth;
        yOffset /= sourceHeight;

        gl.useProgram(program);
        gl.uniform2f(backgroundScaleLocation, xScale, yScale);
        gl.uniform2f(backgroundOffsetLocation, xOffset, yOffset);
    }

    function updateBackgroundImage(backgroundImage, reInit) {
        if (reInit && backgroundTexture) {
            gl.deleteTexture(backgroundTexture);
            backgroundTexture = null;
        }
        if (!backgroundImage) {
            if (backgroundTexture) {
                gl.deleteTexture(backgroundTexture);
                backgroundTexture = null;
            }
            return;
        }

        if (backgroundTexture) {
            gl.bindTexture(gl.TEXTURE_2D, backgroundTexture);
            gl.texSubImage2D(
                gl.TEXTURE_2D,
                0,
                0,
                0,
                backgroundImage.naturalWidth || backgroundImage.videoWidth || backgroundImage.width,
                backgroundImage.naturalHeight || backgroundImage.videoHeight || backgroundImage.height,
                gl.RGBA,
                gl.UNSIGNED_BYTE,
                backgroundImage,
            );
        } else {
            initBackgroundImage(backgroundImage);
        }
    }

    function updateParams(params) {
        gl.useProgram(program);
        const coverage = params.coverage || [0, 1];
        gl.uniform2f(coverageLocation, coverage[0], coverage[1]);

        updateKernel(params.blurKernel || "og10");

        const personColor = parseColorToRGB(params.personColor || "#ffffff", [1, 1, 1]);
        const backgroundColor = parseColorToRGB(params.backgroundColor || "#ffffff", [1, 1, 1]);

        gl.uniform1f(maskFeatherPxLocation, params.maskFeatherPx || 0);
        gl.uniform1i(personModeLocation, params.personMode ?? MODE_NONE);
        gl.uniform1f(personBlurRadiusLocation, params.personBlurRadius || 0);
        gl.uniform1f(personGreyscaleLocation, params.personGreyscale ? 1 : 0);
        gl.uniform3f(personColorLocation, personColor[0], personColor[1], personColor[2]);

        gl.uniform1i(backgroundModeLocation, params.backgroundMode ?? MODE_NONE);
        gl.uniform1f(backgroundBlurRadiusLocation, params.backgroundBlurRadius || 0);
        gl.uniform1f(backgroundGreyscaleLocation, params.backgroundGreyscale ? 1 : 0);
        gl.uniform3f(backgroundColorLocation, backgroundColor[0], backgroundColor[1], backgroundColor[2]);
    }

    function render() {
        gl.viewport(0, 0, outputWidth, outputHeight);
        gl.useProgram(program);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, personMaskTexture);

        gl.uniform1f(useBackgroundTextureLocation, backgroundTexture ? 1 : 0);
        if (backgroundTexture) {
            gl.activeTexture(gl.TEXTURE2);
            gl.bindTexture(gl.TEXTURE_2D, backgroundTexture);
        }

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    updateParams(params);
    if (initialBackgroundImage) initBackgroundImage(initialBackgroundImage);

    function cleanUp() {
        if (backgroundTexture) gl.deleteTexture(backgroundTexture);
        gl.deleteProgram(program);
        gl.deleteShader(fragmentShader);
        gl.deleteShader(vertexShader);
    }

    return {
        render,
        updateParams,
        cleanUp,
        updateBackgroundImage,
    };
}
