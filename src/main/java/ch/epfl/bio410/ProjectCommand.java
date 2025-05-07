package ch.epfl.bio410;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.OpenDialog;
import ij.plugin.GaussianBlur3D;
import ij.plugin.ImageCalculator;
import ij.plugin.ZProjector;
import ij.process.FloatProcessor; // for temporal median filtering
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.util.ArrayList; // for temporal median filtering
import java.util.Arrays; // for temporal median filtering
import java.util.List; // for temporal median filtering


@Plugin(type = Command.class, menuPath = "Plugins>BII>Microtubule Gang")
public class ProjectCommand implements Command {

	public void run() {

		// Prompt the user to select a file
		OpenDialog fileChooser = new OpenDialog("Select a file", null);

		String filePath = fileChooser.getDirectory();
		String fileName = fileChooser.getFileName();
		
		ImagePlus imp = IJ.openImage(filePath+fileName);

		// convert the image to 32 bits for downstream calculations
		IJ.run(imp, "32-bit", "");
		imp.show();

		int nFrames = imp.getNFrames();
		ImageStack outputStack = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int t = 0; t < nFrames; t++) {
			imp.setPosition(1, 1, 1 + t);
			ImageProcessor dog = dog(imp.getProcessor(), 5, 1.25);
			ImageProcessor normed_dog = normalisation(dog);
			outputStack.addSlice(normed_dog);
		}
		// new ImagePlus("DoG Stack", outputStack).show();

		// Enhance contrast on DoG image
		ImagePlus dogProcessedImp = new ImagePlus("Processed DoG Stack", outputStack);
		double max_pixel_value_1 = dogProcessedImp.getStatistics().max ;
		dogProcessedImp.setDisplayRange(0, max_pixel_value_1);
		dogProcessedImp.updateAndDraw();
		dogProcessedImp.show();

		ImagePlus temporalExposure = temporalMaxIntensity(dogProcessedImp, "temporal", 3);
		double max_pixel_value = temporalExposure.getStatistics().max ;
		temporalExposure.setDisplayRange(0, max_pixel_value);
		temporalExposure.updateAndDraw();
		temporalExposure.show();


//		//Apply median filter
//		ImagePlus medianImp = temporalMedianFilter(imp, "Median Stack", 15);
//		// medianImp.show();
//
//    	// Apply dog filter on median filtered image
//		ImageStack dogMedianStack = new ImageStack(medianImp.getWidth(), medianImp.getHeight());
//		for (int t = 0; t < medianImp.getNSlices(); t++) {
//			medianImp.setPosition(1, t+1, t+1);
//			ImageProcessor dog2 = dog(medianImp.getProcessor(), 5, 1.25);
//			ImageProcessor normed_dog2 = normalisation(dog2);
//			dogMedianStack.addSlice(normed_dog2);
//		}
//		// new ImagePlus("DoG on Median Stack", dogMedianStack).show();
//
//		// Enhance contrast on DoG and Median filtered image
//		ImagePlus dogMedianProcessedImp = new ImagePlus("Processed DoG on Median Stack", dogMedianStack);
//		double max_pixel_value_2 = dogMedianProcessedImp.getStatistics().max ;
//		dogMedianProcessedImp.setDisplayRange(0, max_pixel_value_2);
//		dogMedianProcessedImp.updateAndDraw();
//		dogMedianProcessedImp.show();

//		// Apply median filter with simple command
//		ImagePlus median2Imp = imp.duplicate();
//		median2Imp.setTitle("Median 2 Stack");
//		IJ.run(median2Imp, "Median...", "radius=15 stack");
//		ImagePlus medianProcessed2Imp = ImageCalculator.run(imp, median2Imp, "Subtract create stack");
//		//IJ.run(medianProcessed2Imp, "Enhance Contrast", "saturated=0.1 normalize process_all");
//		medianProcessed2Imp.setTitle("Processed Median Stack 2");
//		//medianProcessed2Imp.show();
//
//		// Apply dog filter on median filtered image 2
//		ImageStack dogMedian2Stack = new ImageStack(medianProcessed2Imp.getWidth(), medianProcessed2Imp.getHeight());
//		for (int t = 0; t < medianProcessed2Imp.getNFrames(); t++) {
//			medianProcessed2Imp.setPosition(1, 1, t+1);
//			ImageProcessor dog3 = dog(medianProcessed2Imp.getProcessor(), 5, 1.25);
//			dogMedian2Stack.addSlice(dog3);
//		}
//		//new ImagePlus("DoG on Median Stack 2", dogMedian2Stack).show();
//
//		// Enhance contrast on DoG and Median filtered image 2
//		ImagePlus dogMedian2Imp = new ImagePlus("Processed DoG on Median Stack 2", dogMedian2Stack);
//		IJ.run(dogMedian2Imp, "Enhance Contrast", "saturated=0.35 normalize process_all");
//		dogMedian2Imp.show();

	}

	/**
	 * This method applies a DoG filter on every time frame of the ImagePlus input.
	 * Sigma1 and Sigma2 are unpaired to have better control on the filtering, which
	 * is best suited for our application here.
	 *
	 * @param ip the processor of a given ImagePlus object
	 * @param sigma1 the level of blur of the first gaussian filter = what objects we want to remove (background)
	 * @param sigma2 the level of blur of the second gaussian filter = what objects we want to keep
	 * @return ImageProcessor of the DoG filtered image
	 */
	private ImageProcessor dog(ImageProcessor ip, double sigma1, double sigma2 ) {
		ImagePlus g1 = new ImagePlus("g1", ip.duplicate());
		ImagePlus g2 = new ImagePlus("g2", ip.duplicate());
//		double sigma2 = (Math.sqrt(2) * sigma);
		GaussianBlur3D.blur(g1, sigma1, sigma1, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g2, g1, "Subtract create stack");

		return dog.getProcessor();
	}

	// function to normalise image
	private ImageProcessor normalisation(ImageProcessor ip){
		ImagePlus frame = new ImagePlus("f",ip.duplicate());
		ImageStatistics statistics = frame.getStatistics();
		double std = statistics.stdDev;
		double mean = statistics.mean;
		if (std == 0) std = 1; // Avoid division by zero
		IJ.run(frame, "Subtract...", "value="+mean+" slice");
		IJ.run(frame, "Divide...", "value="+std+" slice");

////			// Apply contrast enhancement again
//		ce.stretchHistogram(inFocusNorm[t].getProcessor(), 0.35);
//		frame.setDisplayRange(-4.30, 8.11);
//		frame.updateAndDraw();
//			IJ.run(inFocusNorm[t], "Apply LUT", "slice");
		return frame.getProcessor();
	}

	private ImagePlus temporalMaxIntensity(ImagePlus imp, String title, int window){
		int nFrames = imp.getNSlices();
		ImagePlus copy = imp.duplicate();

		ImageStack results = new ImageStack(imp.getWidth(), imp.getHeight());

		for(int t=1; t<= nFrames; t++){
			copy.setPosition(1,1, t);
			IJ.log("in loop"+t);
			// Edge Conditions
			if(t-window < 1){
				ImageProcessor ip = ZProjector.run(copy,"max",t,t+window).getProcessor();
				results.addSlice(ip);
				IJ.log("start"+t);
			} else if (t+window > nFrames){
				ImageProcessor ip = ZProjector.run(copy,"max",t-window,t).getProcessor();
				results.addSlice(ip);
				IJ.log("end"+t);
			} else {
				ImageProcessor ip = ZProjector.run(copy,"max",t-window,t+window).getProcessor();
				results.addSlice(ip);
				IJ.log("midlle"+t);
//				results = ZProjector.run(copy, "max",t-window, t+window);
			}
		}

		ImagePlus resultImp = new ImagePlus(title, results);
//		resultImp.setDimensions(1, 1, nFrames);  // 1 channel, 1 slice, nFrames time points

		return  resultImp;
	}
	/**
	 * This method applies a temporal median filter to a 32-bit time series.
	 * For each frame in the time series, the function collects pixel values from neighboring
	 * frames within a specified radius, computes the median per pixel across this temporal
	 * window, and subtracts the median from the current frame.
	 * Only single-slice, single-channel, 32-bit time series are supported.
	 *
	 * @param imp     The ImagePlus input expected to be a 32-bit image with one slice, one channel and multiple frames
	 * @param title   Title of the output image
	 * @param radius  Radius of the temporal window in frames (e.g. a radius of 4 uses 9 frame if available)
	 * @return A new ImagePlus containing the temporally filtered frames
	 * @throws IllegalArgumentException if the input is not 32-bit float, or has more than one slice or channel
	 */
	private ImagePlus temporalMedianFilter(ImagePlus imp, String title, int radius) {
		// Throw errors the image given is not 32-bit and if there are more than one slice or channels
		if (imp.getType() != ImagePlus.GRAY32)
			throw new IllegalArgumentException("Input must be a 32-bit float image stack.");
		if (imp.getNChannels() != 1 || imp.getNSlices() != 1)
			throw new IllegalArgumentException("Only single-slice, single-channel time series supported.");

		int w = imp.getWidth(), h = imp.getHeight();
		int nFrames = imp.getNFrames();
		ImageStack stack = imp.getStack().duplicate();
		ImageStack result = new ImageStack(w, h);

		// Loop over time frames to collect frames in temporal window
		for (int t = 1; t <= nFrames; t++) {
			List<float[]> window = new ArrayList<>();
			for (int dt = -radius; dt <= radius; dt++) {
				int ti = t + dt;
				if (ti >= 1 && ti <= nFrames) {
					int index = imp.getStackIndex(1, 1, ti);
					window.add((float[]) stack.getProcessor(index).getPixels()); // store frame pixel array
				}
			}

			float[] input = (float[]) stack.getProcessor(imp.getStackIndex(1, 1, t)).getPixels();
			float[] output = new float[w * h];

			// Pixel-wise median subtraction
			for (int i = 0; i < w * h; i++) {
				float[] values = new float[window.size()];
				for (int j = 0; j < window.size(); j++) values[j] = window.get(j)[i];
				Arrays.sort(values);
				float median = values[values.length / 2];
				output[i] = input[i] - median; // here chatgpt advised clipping results to have non-negative values
				// but I don't see the point in our case : it was done with this command: Math.max(0, input[i] - median)
			}
			result.addSlice(new FloatProcessor(w, h, output)); // store the filtered frame
		}
		return new ImagePlus(title, result);
	}

	/**
	 * This main function serves for development purposes.
	 * It allows you to run the plugin immediately out of
	 * your integrated development environment (IDE).
	 *
	 * @param args whatever, it's ignored
	 * @throws Exception
	 */
	public static void main(final String... args) throws Exception {
		final ImageJ ij = new ImageJ();
		ij.ui().showUI();
	}
}
