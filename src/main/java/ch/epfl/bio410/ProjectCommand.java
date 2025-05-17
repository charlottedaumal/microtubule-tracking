package ch.epfl.bio410;

import ch.epfl.bio410.cost.AbstractDirCost;
import ch.epfl.bio410.cost.DirectionCost;
import ch.epfl.bio410.graph.PartitionedGraph;
import ch.epfl.bio410.graph.Spot;
import ch.epfl.bio410.graph.Spots;
import ch.epfl.bio410.utils.TemporalDifferencer;
import ch.epfl.bio410.utils.TemporalProjector;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.gui.GenericDialog;
import ij.io.OpenDialog;
import ij.plugin.GaussianBlur3D;
import ij.plugin.ImageCalculator;
import ij.plugin.ZProjector;
import ij.process.Blitter;
import ij.process.FloatProcessor; // for temporal median filtering
import ij.process.ImageProcessor;
import ij.process.ImageStatistics;
import net.imagej.ImageJ;
import org.ojalgo.access.StreamAnyD;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.awt.*;
import java.util.ArrayList; // for temporal median filtering
import java.util.Arrays; // for temporal median filtering
import java.util.List; // for temporal median filtering
import java.util.Objects;


@Plugin(type = Command.class, menuPath = "Plugins>BII>Microtubule Gang")
public class ProjectCommand implements Command {

	private double costmax=0.5;
	private double sigma ;  // sigma of the DoG TODO : adapt it
	private double sigma1 ; // sigma1 of preprocessing : to remove background
	private double sigma2 ; // sigma2 of preprocessing : what we want to keep from image
	private int windowExp ; // size of window to increase exposure
	private int windowDiff ; // size of window to delete continuous tracks
	private double threshold ; // threshold of intensity to detect spots
	private double tolerance ; // tolerance of spots detected around one spot
	private double lambda ; // wieght of distance in cost computation
	private double gamma ; // weight of direction in cost computation
	private double kappa ;

	public void run() {

		// Prompt the user to select a file
		OpenDialog fileChooser = new OpenDialog("Select a file", null);

		String filePath = fileChooser.getDirectory();
		String fileName = fileChooser.getFileName();
		
		ImagePlus imp = IJ.openImage(filePath+fileName);

		GenericDialog gd = new GenericDialog("Selecting the parameters");

		// user input parameters
		gd.addNumericField("Costmax:", 0.5, 3);
		gd.addNumericField("Sigma:", 1, 2);
		gd.addNumericField("Sigma1:", 5,2);
		gd.addNumericField("Sigma2:", 1.25,2);
		gd.addNumericField("WindowExp", 3, 1);
		gd.addNumericField("WindowDiff", 1, 1);
		gd.addNumericField("Threshold", 10, 3);
		gd.addNumericField("Tolerance", 0, 1);
		gd.addNumericField("Lambda", 0.5, 3);
		gd.addNumericField("Gamma", 0.3, 3);
		gd.addNumericField("Kappa", 0.15, 3);


		gd.showDialog();

		// === Retrieve the values from GUI ===
		costmax = gd.getNextNumber();
		sigma = gd.getNextNumber();
		sigma1 = gd.getNextNumber();
		sigma2 = gd.getNextNumber();
		windowExp = (int) gd.getNextNumber();
		windowDiff = (int) gd.getNextNumber();
		threshold = gd.getNextNumber();
		tolerance = gd.getNextNumber();
		lambda = gd.getNextNumber();
		gamma = gd.getNextNumber();
		kappa = gd.getNextNumber();

		// convert the image to 32 bits for downstream calculations
		IJ.run(imp, "32-bit", "");
		//imp.show();

		int nFrames = imp.getNFrames();

//		ImagePlus outputstack = new ImagePlus("Normalised modified Dog", normalisation((dog(imp.getProcessor(), sigma1, sigma2))));
		ImagePlus result = processStack(imp, ip -> dog(ip, sigma1, sigma2));
		ImagePlus outputstack = processStack(result, this::normalisation);

//		for (int t = 0; t < nFrames; t++) {
//			imp.setPosition(1, 1, 1 + t);
//			ImageProcessor dog = dog(imp.getProcessor(), 5, 1.25);
//			ImageProcessor normed_dog = normalisation(dog);
//			outputStack.addSlice(normed_dog);
//		}
		// new ImagePlus("DoG Stack", outputStack).show();

		// Enhance contrast on DoG image
//		ImagePlus dogProcessedImp = new ImagePlus("Processed DoG Stack", outputStack);
		double max_pixel_value_1 = outputstack.getStatistics().max ;
//		outputstack.setDisplayRange(0, max_pixel_value_1);
//		outputstack.updateAndDraw();
		IJ.run(outputstack, "Median...", "radius=1 stack");
		outputstack.show();

		// ---- temporal projection using the wrapper ----
		TemporalProjector tp = new TemporalProjector(outputstack,
							"sum",     // projection type
									"left",  // window place
									windowExp);        // window size
		ImagePlus temporalExposure = processStack(outputstack, (ImageProcessor ip) -> tp.apply(ip));

//		ImagePlus temporalExposure = temporalProjection(outputstack, "temporal", "sum",
//				"left",windowExp);
//		double max_pixel_value = temporalExposure.getStatistics().max ;
//		temporalExposure.setDisplayRange(0, max_pixel_value);
//		temporalExposure.updateAndDraw();
//		temporalExposure.show();

//		ImagePlus tempDiff = temporalDifference(temporalExposure,"difference",windowDiff);
		TemporalDifferencer diff = new TemporalDifferencer(temporalExposure, windowDiff);
		ImagePlus tempDiff = processStack(imp, (ImageProcessor ip) -> diff.apply(ip));

//		tempDiff.setDisplayRange(0, tempDiff.getStatistics().max );
//		tempDiff.updateAndDraw();
		tempDiff.show();

		PartitionedGraph framesDiff = detect(tempDiff,1,5,tolerance);
		framesDiff.drawSpots(tempDiff);
		AbstractDirCost cost = new DirectionCost(outputstack, costmax, lambda, gamma, kappa);

		int dimension = 20;
//		PartitionedGraph trajectories = trackToFirstValidTrajectory(frames, cost);
		PartitionedGraph trajectoriesDiff = directionalTracking(framesDiff, cost, dimension);
		PartitionedGraph cleanTraj = cleaningTrajectories(trajectoriesDiff, 5);
		cleanTraj.drawLines(tempDiff);

//		// Detection
//		PartitionedGraph frames = detect(temporalExposure, sigma, threshold, tolerance);
//		frames.drawSpots(temporalExposure);
//
////		PartitionedGraph trajectories = trackToFirstValidTrajectory(frames, cost);
//		PartitionedGraph trajectories = directionalTracking(frames, cost, dimension);
//		trajectories.drawLines(temporalExposure);

//		ImagePlus total_proj = totalProjection(outputstack, "max");
//		total_proj.show();



//
//		double costmax=5;
//		double lambda = 0.99;
//		AbstractCost cost = new DistanceAndIntensityCost(imp, costmax, lambda);


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


	/**
	 * This method normalises the image pixels values for a single processor.
	 *
	 * @param ip the image processor to normalise
	 * @return ImageProcessor normalised
	 */
	private ImageProcessor normalisation(ImageProcessor ip){
		ImagePlus frame = new ImagePlus("f",ip.duplicate());
		ImageStatistics statistics = frame.getStatistics();
		double std = statistics.stdDev;
		double mean = statistics.mean;
		if (std == 0) std = 1; // Avoid division by zero
		IJ.run(frame, "Subtract...", "value="+mean+" slice");
		IJ.run(frame, "Divide...", "value="+std+" slice");

		return frame.getProcessor();
	}

	@FunctionalInterface
	interface ImageProcessorFunction {
		ImageProcessor apply(ImageProcessor ip);
	}


	/**
	 * This method allows to process and apply the given function to a whole ImagePlus object.
	 *
	 * @param imp The ImagePlus on which we want to apply the function
	 * @param func The function that we want to apply to each frame's processor
	 * @return A new ImagePlus object where func has been applied to each image processor
	 */
	private ImagePlus processStack(ImagePlus imp, ImageProcessorFunction func) {
		ImageStack newStack = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int i = 1; i <= imp.getStackSize(); i++) {
			ImageProcessor ip = imp.getStack().getProcessor(i);
			ImageProcessor result = func.apply(ip);  // Call passed function
			newStack.addSlice(result);
			System.gc();  // Encourage cleanup

		}

		ImagePlus resultImp = new ImagePlus("Processed", newStack);
		resultImp.setDimensions(1, 1, imp.getStackSize());
		resultImp.setOpenAsHyperStack(true);
		return resultImp;
	}

	/**
	 * This method computes the desired intensity projection on the t axis on the ImagePlus input
	 * to increase contrast and exposure time of the original image. It is done so by a sliding window
	 * that takes the total amount of frames 2*window + 1 centered around the considered frame and sums them.
	 *
	 * @param imp image input
	 * @param title title we want to give to the results
	 * @param typeOfProjection the type of projection we want, can be "max", "min, "sum", "sd", etc
	 * @param window_place either 'left', 'middle' or 'right' determines the location of the window with respect to the frame
	 * @param window number of frames/2 that we want to be projected
	 * @return ImagePlus where the sliding window projection has been done
	 */
	private ImagePlus temporalProjection(ImagePlus imp, String title, String typeOfProjection, String window_place, int window){
		int nFrames = imp.getNFrames();
		ImagePlus copy = imp.duplicate();

		ImageStack results = new ImageStack(imp.getWidth(), imp.getHeight());

		for(int t=1; t<= nFrames; t++){
			copy.setPosition(1,1, t);

			if(Objects.equals(window_place, "middle")){
				int start = Math.max(1, t - window/2);
				int end = Math.min(nFrames, t + window/2);
				// Run temporal max projection
				ImageProcessor ip = ZProjector.run(copy, typeOfProjection, start, end).getProcessor();
				results.addSlice("Frame " + t, ip);
			} else if (Objects.equals(window_place, "left")) {
				int start = Math.max(1, t - window);
				// Run temporal max projection
				ImageProcessor ip = ZProjector.run(copy, typeOfProjection, start, t).getProcessor();
				results.addSlice("Frame " + t, ip);
			}else if (Objects.equals(window_place, "right")){
				int end = Math.min(nFrames, t + window);
				// Run temporal max projection
				ImageProcessor ip = ZProjector.run(copy, typeOfProjection, t, end).getProcessor();
				results.addSlice("Frame " + t, ip);
			}
		}
		// Create the final ImagePlus and treat it as a time series
		ImagePlus resultImp = new ImagePlus(title, results);
		resultImp.setDimensions(1, 1, nFrames);        // 1 channel, 1 Z slice, n time frames
		resultImp.setOpenAsHyperStack(true);

		return  resultImp;
	}

	/**
	 * TODO finish docstring
	 * This method computes the desired intensity projection on the t axis on the ImagePlus input
	 * to increase contrast and exposure time of the original image. It is done so by a sliding window
	 * that takes the total amount of frames 2*window + 1 centered around the considered frame and sums them.
	 *
	 * @param imp image input
	 * @param typeOfProjection the type of projection we want, can be "max", "min, "sum", "sd", etc
	 * @param window_place either 'left', 'middle' or 'right' determines the location of the window with respect to the frame
	 * @param window number of frames/2 that we want to be projected
	 * @return ImagePlus where the sliding window projection has been done
	 */
	private ImageProcessor temporalProjectionFrame(ImagePlus imp, int t, String typeOfProjection, String window_place, int window){
		int nFrames = imp.getNFrames();
		ImagePlus copy = imp.duplicate();

		ImageStack results = new ImageStack(imp.getWidth(), imp.getHeight());

		copy.setPosition(1,1, t);
		int start= 0;
		int end= 0;

		if(Objects.equals(window_place, "middle")){
			start = Math.max(1, t - window/2);
			end = Math.min(nFrames, t + window/2);

		} else if (Objects.equals(window_place, "left")) {
			start = Math.max(1, t - window);
			end = t;
		}else if (Objects.equals(window_place, "right")){
			start = t;
			end = Math.min(nFrames, t + window);
		}

		for (int i =start; i<=end; i++){
			imp.setPosition(1,1,i);
			results.addSlice(imp.getProcessor().duplicate());
		}

		ImagePlus temp = new ImagePlus("temp", results);
		ImageProcessor projection = ZProjector.run(temp, typeOfProjection).getProcessor();
		temp.close();

		return  projection;
	}

	private ImagePlus totalProjection(ImagePlus imp, String typeOfProjection){
		return ZProjector.run(imp,typeOfProjection);
	}

	private ImagePlus temporalDifference(ImagePlus imp, String title, int windowSize) {
		int nFrames = imp.getNFrames();
		ImageStack results = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int t = 1; t <= nFrames; t++) {
			int start = Math.max(1, t - windowSize);
			int end = t;

			imp.setPosition(1, 1, t);
			ImageProcessor current = imp.getProcessor().duplicate();
			for(int i = start; i <end; i++){
				imp.setPosition(1, 1, i);
				ImageProcessor previous = imp.getProcessor().duplicate();
				current.copyBits(previous, 0, 0, Blitter.SUBTRACT);
				previous = null; // freeing up memory
			}

			results.addSlice("Δ Frame " + t, current);
			current = null ; // freeing up memory
			System.gc();  // Encourage cleanup

		}

		ImagePlus resultImp = new ImagePlus(title, results);
		resultImp.setDimensions(1, 1, nFrames );
		resultImp.setOpenAsHyperStack(true);

		return resultImp;
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
	 * TODO question 1 - fill the method description and input/output parameters
	 * This method applies a DoG filter on every time frame of the ImagePlus input.
	 * Then every spot on each frame is detected if the intensity is above the specified threshold.
	 * The size of the spots detected is saved into the IJ log and they are added into a partitioned graph
	 * which is returned by the method.
	 *
	 * @param imp the ImagePlus object input
	 * @param sigma the level of blur of the gaussian filter of the DoG
	 * @param threshold the threshold of intensity to detect spots in the image input imp
	 * @return Graph that contains the spots detected
	 */
	private PartitionedGraph detect(ImagePlus imp, double sigma, double threshold, double tolerance) {

		int nt = imp.getNFrames();
		new ImagePlus("DoG", classic_dog(imp.getProcessor(), sigma)).show();
		PartitionedGraph graph = new PartitionedGraph();
		for (int t = 0; t < nt; t++) {
			imp.setPosition(1, 1, 1+t);
			ImageProcessor ip = imp.getProcessor();
			ImageProcessor dog = classic_dog(ip, sigma);
			Spots spots = localMax(dog, ip, t, threshold, tolerance);
			IJ.log("Frame t:" + t + " #localmax:" + spots.size() );
			graph.add(spots);
		}
		return graph;
	}

	/**
	 * This method creates a DoG filter processor for a given processor.
	 * The level of blur of the filter is adjusted by the parameter sigma.
	 *
	 * @param ip the ImageProcessor of an ImagePlus object
	 * @param sigma the level of blur of the gaussian filters
	 * @return a DoG processor
	 */
	private ImageProcessor classic_dog(ImageProcessor ip, double sigma) {
		ImagePlus g1 = new ImagePlus("g1", ip.duplicate());
		ImagePlus g2 = new ImagePlus("g2", ip.duplicate());
		double sigma2 = (Math.sqrt(2) * sigma);
		GaussianBlur3D.blur(g1, sigma, sigma, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g1, g2, "Subtract create stack");
		return dog.getProcessor();
	}

	public Spots localMax(ImageProcessor dog, ImageProcessor image, int t, double threshold, double tolerance) {
		Spots spots = new Spots();
		// going through the image pixel by pixel
		for (int x = 1; x < dog.getWidth() - 1; x++) {
			for (int y = 1; y < dog.getHeight() - 1; y++) {
				double valueImage = image.getPixelValue(x, y);
				// compare value of the pixel to the threshold
				if (valueImage >= threshold) {
					// if above the threshold we get the corresponding pixel value after applying the DoG filter
					double v = dog.getPixelValue(x, y);
					double max = -1;
					// check neighbor pixels of the detected pixel above the threshold
					for (int k = -1; k <= 1; k++) // k=-1,0,1
						for (int l = -1; l <= 1; l++) // l=-1,0,1
							// we save the maximum value between the 9 pixels centered around the pixel above the threshold found
							max = Math.max(max, dog.getPixelValue(x + k, y + l));
					// if the pixel in the center is the max between the 9 pixels then we add a Spot to the list
					if (v == max) spots.add(new Spot(x, y, t, valueImage));
				}
			}
		}

//		Spots final_spots = new Spots();
//		for (Spot x : spots){
//			for (Spot y : spots) {
//				if(! x.equals(y)){
//					if (x.distance(y) < tolerance) {
//						IJ.log("in tolerance loop"+x.distance(y));
//						if (x.value < y.value){
//							final_spots.add(y);
//						}else{
//							final_spots.add(x);
//						}
//					}
//				}
//
//			}
//		}
//		return final_spots; // return the final list of Spots
		return spots;
	}


	private PartitionedGraph directionalTracking(PartitionedGraph frames, AbstractDirCost cost, int dimension) {
		PartitionedGraph trajectories = new PartitionedGraph();
		for (Spots frame : frames) {
			for (Spot spot : frame) {
				Spots trajectory = trajectories.getPartitionOf(spot);
				if (trajectory == null) trajectory = trajectories.createPartition(spot);
				if (spot.equals(trajectory.last())) {
					int t0 = spot.t;
					// TODO question 4 - add trajectory to the nearest spot of the next frame
					for (int t=t0; t < frames.size() - 1; t++) {
						double trajectory_cost = this.costmax; // set the first cost value to be the highest possible
						Spot next_spot = null; // we first initialize the next_spot to be null
						for(Spot next : frames.get(t+1)) { // iterate over all spots of the next frame
							if (cost.validate(next, spot, frames, dimension) == true) { // if the cost is lesser than the costmax
								if(cost.evaluate(next, spot, frames, dimension) < trajectory_cost) {
									// if the new cost is less than the previous one we save the spot
									trajectory_cost = cost.evaluate(next, spot, frames, dimension);
									next_spot = next;
								}
							}
						}
						if (next_spot != null) { // check that we found a next spot to add to the trajectory
							// after iteration over all spots on next frame, final spot is saved in next spot
							IJ.log("#" + trajectories.size() + " spot " + next_spot + " with a cost:" + trajectory_cost);
							trajectory.add(next_spot);
							spot = next_spot;
						} else {
							break;  // no valid match found, stop this trajectory
						}
					}
				}
			}
		}
		return trajectories;
	}

	private PartitionedGraph cleaningTrajectories(PartitionedGraph frames, int min_length){
		// trying to clean up minimal trajectories to lighten memory load
		PartitionedGraph final_graph = new PartitionedGraph();
		for (Spots trajectory : frames){
			if (trajectory.size() > min_length){
				final_graph.add(trajectory);
			}
		}
		return final_graph;
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
