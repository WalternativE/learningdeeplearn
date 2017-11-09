import { Array3D, gpgpu_util, GPGPUContext, NDArrayMathCPU, NDArrayMathGPU } from 'deeplearn';
import { SqueezeNet } from 'deeplearn-squeezenet';

export class ImageInferenceDemo {
  private inferenceCanvas: HTMLCanvasElement;
  private imageToInfer: HTMLImageElement;

  private math: NDArrayMathGPU;
  private mathCPU: NDArrayMathCPU;
  private gl: WebGLRenderingContext;
  private gpgpu: GPGPUContext;

  private squeezeNet: SqueezeNet;

  private layerNames: string[];

  constructor() {
    this.inferenceCanvas = document.getElementById('inference-canvas') as HTMLCanvasElement;
    this.imageToInfer = document.getElementById('image-to-infer') as HTMLImageElement;

    this.gl = gpgpu_util.createWebGLContext(this.inferenceCanvas);
    this.gpgpu = new GPGPUContext(this.gl);
    this.math = new NDArrayMathGPU(this.gpgpu);
    this.mathCPU = new NDArrayMathCPU();

    this.squeezeNet = new SqueezeNet(this.math);
    this.squeezeNet.load().then(() => { this.infer() });

    this.layerNames = [];

    console.log('Demo loaded');
  }

  async infer() {
    await this.math.scope(async (keep, track) => {
      const image = track(Array3D.fromPixels(this.imageToInfer));
      console.log(image);

      const inferenceResult = await this.squeezeNet.predict(image);
      console.log(inferenceResult);
      const namedActivations = inferenceResult.namedActivations;

      this.layerNames = Object.keys(namedActivations);
      console.log(this.layerNames);

      const topClassesToProbability = await this.squeezeNet.getTopKClasses(
        inferenceResult.logits, 5);

      console.log(topClassesToProbability);
    });
  }
}

window.onload = () => {
  const demo = new ImageInferenceDemo();
}