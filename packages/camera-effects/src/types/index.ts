interface Pipeline {
    module: {
        createEffectStream: (
            inputStream: MediaStream,
            setup: Pipeline["getDefaultSetup"],
            params: Params,
        ) => Promise<{
            stream: MediaStream;
            stop: () => void;
            updateParams: (params: Params) => Promise<void> | Promise<boolean>;
            // Feeds the effect from another camera, keeping the output track. Not with insertable streams.
            setInputStream?: (inputStream: MediaStream) => Promise<void>;
        }>;
    };
    getDefaultSetup: () => {
        useWebGL?: boolean;
        segmentationModelId?: string;
        logToConsole?: boolean;
        framerate?: number;
        useInsertableStreams?: boolean;
        canvasMemoryLeakWorkaround?: boolean;
        transferType?: string;
        useBackgroundWorker?: boolean;
        // Where the avatar's face landmarker loads from. Defaults to jsdelivr and Google's model storage.
        faceLandmarkerWasmUrl?: string;
        faceLandmarkerModelUrl?: string;
    };
    canUse: (options: { allowSafari?: boolean }) => boolean;
}

type EffectAmount = "slight" | "normal" | "heavy";

// An effect on one region of the frame, from the segmentation mask.
type LayerEffect =
    | {
          type: "blur";
          amount: EffectAmount;
          greyscale?: boolean;
      }
    | {
          type: "fill";
          color: string;
      };

// Independent effects per layer. A layer left out shows the camera as is; no layers at all passes
// the camera through untouched.
interface LayerParams {
    person?: LayerEffect;
    background?: LayerEffect;
    // Drawn over the other layers. Only the canvas engine draws it.
    avatar?: {
        type: "wireframe";
        color?: string;
    };
}

interface Params {
    // Takes precedence over backgroundUrl, backgroundBlur, anonymization and avatar.
    layers?: LayerParams;
    backgroundUrl?: Promise<unknown> | string;
    backgroundType?: "image" | "video";
    backgroundBlur?: {
        amount: "slight" | "normal" | "heavy";
    };
    anonymization?: {
        type: "pixelation" | "blur" | "silhouette" | "color";
        applyBackground: boolean;
        amount?: "slight" | "normal" | "heavy";
        greyscale?: boolean;
        color?: string;
    };
    avatar?: {
        type: "wireframe" | "2d" | "3d";
        color ?: string;
        modelUrl?: Promise<unknown> | string;
    };
}

interface Preset {
    id: string;
    backgroundUrl?: Promise<string>;
    pipelineConfigs?: {
        [pipelineId: string]: {
            setup?: Pipeline["getDefaultSetup"] | undefined;
            params: Params;
        };
    };
}

export type { Pipeline, Params, Preset, EffectAmount, LayerEffect, LayerParams };
