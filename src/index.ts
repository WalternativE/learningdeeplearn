import { Array3D, gpgpu_util, GPGPUContext, NDArrayMathCPU, NDArrayMathGPU } from 'deeplearn';
import { SqueezeNet } from 'deeplearn-squeezenet';

export class ImageInferenceDemo {
  private _inferenceCanvas: HTMLCanvasElement;
  private _imageToInfer: HTMLImageElement;

  private _math: NDArrayMathGPU;
  private _mathCPU: NDArrayMathCPU;
  private _gl: WebGLRenderingContext;
  private _gpgpu: GPGPUContext;

  private _squeezeNet: SqueezeNet;

  private _isNetLoaded: boolean;

  get isNetLoaded(): boolean { return this._isNetLoaded; }
  get image(): HTMLImageElement { return this._imageToInfer; }

  constructor() {
    this._isNetLoaded = false;

    this._inferenceCanvas = document.getElementById('inference-canvas') as HTMLCanvasElement;
    this._imageToInfer = document.getElementById('image-to-infer') as HTMLImageElement;

    this._gl = gpgpu_util.createWebGLContext(this._inferenceCanvas);
    this._gpgpu = new GPGPUContext(this._gl);
    this._math = new NDArrayMathGPU(this._gpgpu);
    this._mathCPU = new NDArrayMathCPU();

    this._squeezeNet = new SqueezeNet(this._math);
    this._squeezeNet.load().then(() => {
      console.log('Demo loaded');
      this._isNetLoaded = true;
    });
  }

  async infer() {
    await this._math.scope(async (keep, track) => {
      const image = track(Array3D.fromPixels(this._imageToInfer));
      const inferenceResult = await this._squeezeNet.predict(image);
      const topClassesToProbability = await this._squeezeNet.getTopKClasses(
        inferenceResult.logits, 5);

      const displayTarget = document.getElementById('classifier-target') as HTMLDivElement;
      while(displayTarget.hasChildNodes()) {
        displayTarget.removeChild(displayTarget.lastChild);
      }

      const ul = document.createElement("ul");
      Object.keys(topClassesToProbability).forEach(label => {
        let entry = document.createElement("li");
        entry.innerHTML = label + " Prob: " + topClassesToProbability[label];
        ul.appendChild(entry);
      });
      displayTarget.appendChild(ul);

      console.log(topClassesToProbability);
    });
  }
}

window.onload = () => {
  const demo = new ImageInferenceDemo();

  const imgInput = document.getElementById('image-url') as HTMLInputElement;
  const updateButton = document.getElementById('url-update-button') as HTMLButtonElement;

  const handler = () => {
    if (demo.isNetLoaded) {
      updateButton.innerHTML = "Update";
      updateButton.disabled = false;
    } else {
      console.log("give it a sec");
      setTimeout(handler, 500);
    }
  };

  handler();

  updateButton.onclick = (ev) => {
    updateButton.disabled = true;
    updateButton.innerHTML = "Processing";

    demo.image.onload = () => {
      demo.infer()
        .then(() => {
          updateButton.disabled = false;
          updateButton.innerHTML = "Update";
        })
        .catch((reason) => {
          updateButton.disabled = false;
          console.log(reason);
        });
    };
    demo.image.src = imgInput.value;
  };
}