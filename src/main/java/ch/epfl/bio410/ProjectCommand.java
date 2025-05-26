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
import ij.gui.DialogListener;
import ij.gui.GenericDialog;
import ij.gui.NonBlockingGenericDialog;
import ij.gui.Plot;
import ij.io.OpenDialog;
import ij.measure.Calibration;
import ij.plugin.*;
import ij.process.*;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.util.*;
import java.util.List;


@Plugin(type = Command.class, menuPath = "Plugins>BII>MiTrack")
public class ProjectCommand implements Command {

	private double costmax = 0.5;
	private double sigma;  // sigma of the DoG
	private double sigma1; // sigma1 of preprocessing : to remove background
	private double sigma2; // sigma2 of preprocessing : what we want to keep from image
	private int windowDiff = 1; // size of window to delete continuous tracks
	private double threshold; // threshold of intensity to detect spots
	private double lambda; // weight of distance in cost computation
	private double gamma; // weight of direction in cost computation
	private double kappa;

	public void run() {

		// prompt the user to select a file
		OpenDialog fileChooser = new OpenDialog("Select a file", null);
		String filePath = fileChooser.getDirectory();
		String fileName = fileChooser.getFileName();
		
		ImagePlus imp = IJ.openImage(filePath+fileName);

		GenericDialog gd = new GenericDialog("Welcome to the MiTrack Plugin :)");

		// parameter tuning part I section
		gd.addMessage("=== Parameter Tuning Part I ===");

		String[] displayOptions = {
				"Do not display",
				"Display",
		};

		// preprocessing
		gd.addMessage(">> Preprocessing <<");
		gd.addNumericField("Sigma1:", 5,2);
		gd.addToSameRow();
		gd.addNumericField("Sigma2:", 1.25,2);
		gd.addMessage("");

		// slider for WindowExp
		int windowExpInit = 3; // initial value
		int minExp = 1;
		int maxExp = 3;
		final int[] windowExp = {windowExpInit}; // use array to allow mutation in lambda

		// Create JSlider
		JSlider slider = new JSlider(JSlider.HORIZONTAL, minExp, maxExp, windowExpInit);
		slider.setMajorTickSpacing(1);
		slider.setPaintTicks(true);
		slider.setPaintLabels(true);

		// Label to show current value
		JLabel valueLabel = new JLabel("     WindowExp:    ");
		// Update value on slider move
		slider.addChangeListener(e -> {
			windowExp[0] = slider.getValue();
		});

		// wrap slider and label in AWT Panel
		Panel sliderPanel = new Panel(new BorderLayout());
		sliderPanel.add(slider, BorderLayout.CENTER);
		sliderPanel.add(valueLabel, BorderLayout.WEST);
		gd.addPanel(sliderPanel);

		gd.addMessage("");
		gd.addChoice("Preprocessed Stack", displayOptions, displayOptions[0]);

		gd.showDialog();
		if (gd.wasCanceled()) return;

		// retrieve the values from the first GUI
		sigma1 = gd.getNextNumber();
		sigma2 = gd.getNextNumber();
		int chosenWindowExp = windowExp[0];
		String choicePreprocessed = gd.getNextChoice();

		// convert the image to 32 bits for downstream calculations
		IJ.run(imp, "32-bit", "");
		imp.setTitle("Original Image");
		imp.show();

		// applying dog filter and median filtering for denoizing and background homogenization
		ImagePlus result = processStack(imp, ip -> dog(ip, sigma1, sigma2));
		ImagePlus outputstack = processStack(result, this::normalisation);
		outputstack.setTitle("Preprocessed Stack");
		IJ.run(outputstack, "Median...", "radius=1 stack");

		// display the preprocessed stack depending on the choice of the user
		switch (choicePreprocessed) {
			case "Do not display":
				break;
			case "Display":
				outputstack.setDisplayRange(0, outputstack.getStatistics().max);
				outputstack.updateAndDraw();
				outputstack.show();
				break;
		}

		// temporal projection
		TemporalProjector tp = new TemporalProjector(outputstack, "sum", "left", chosenWindowExp);
		ImagePlus temporalExposure = processStack(outputstack, (ImageProcessor ip) -> tp.apply(ip));

		TemporalDifferencer diff = new TemporalDifferencer(temporalExposure, windowDiff);
		ImagePlus tempDiff = processStack(temporalExposure, (ImageProcessor ip) -> diff.apply(ip));
		tempDiff.setTitle("Temporal Projection");
		tempDiff.setDisplayRange(0, tempDiff.getStatistics().max);
		tempDiff.updateAndDraw();
		tempDiff.show();

		NonBlockingGenericDialog gd2 = new NonBlockingGenericDialog("Select further parameters!");

		// parameter tuning part II section
		gd2.addMessage("=== Parameter Tuning Part II ===");

		// segmentation
		gd2.addMessage(">> Segmentation <<");
		gd2.addNumericField("Sigma:", 1, 2);
		gd2.addToSameRow();
		gd2.addNumericField("Threshold", 5, 3);


		// add preview checkbox for the threshold
		gd2.addCheckbox("Preview detection", false);

		gd2.addDialogListener((dialog, e) -> {
			if (gd2.wasCanceled()) return false;
			Checkbox previewCheckbox = (Checkbox) gd2.getCheckboxes().get(0);
			if (previewCheckbox.getState()) {
				double previewSigma = Double.parseDouble(((TextField) gd2.getNumericFields().get(0)).getText());
				double previewThreshold = Double.parseDouble(((TextField) gd2.getNumericFields().get(1)).getText());
				int frame = chosenWindowExp + 2;
				ImagePlus singleFrame = new ImagePlus("Frame " + (frame),
						tempDiff.getStack().getProcessor(frame).duplicate());
				singleFrame.setDimensions(1, 1, 1);

				PartitionedGraph preview = detect(singleFrame, previewSigma, previewThreshold, false);
				preview.drawSpots(singleFrame);
				previewCheckbox.setState(false);
			}
			return true;
		});

		// tracking
		gd2.addMessage(">> Tracking <<");
		gd2.addNumericField("Costmax:", 0.5, 3);
		gd2.addToSameRow();
		gd2.addNumericField("Lambda", 0.5, 3);
		gd2.addNumericField("Gamma", 0.3, 3);
		gd2.addToSameRow();
		gd2.addNumericField("Kappa", 0.15, 3);
		gd2.addMessage("");

		// results options
		gd2.addMessage("=== Results options ===");

		String[] coloringOptions = {
				"Random",
				"Global average orientation",
				"Average local orientation",
				"Instantaneous velocity",
		};
		String[] costOptions = {
				"Balanced with distance, direction and intensity",
				"Balanced with distance, direction, intensity and speed",
		};
		String[] legendOptions = {
				"Do not display",
				"Legend for orientation coloring",
				"Legend for velocity coloring",
		};

		// trajectories determination and display
		gd2.addChoice("Cost function", costOptions, costOptions[0]);
		gd2.addChoice("Coloring of Trajectories", coloringOptions, coloringOptions[0]);
		gd2.addToSameRow();
		gd2.addChoice("Color map legend", legendOptions, legendOptions[0]);

		// speeds
		gd2.addCheckbox("Average speed distribution", false);
		gd2.addToSameRow();
		gd2.addCheckbox("Speed evolution from TopN longest trajectories", false);
		gd2.addToSameRow();
		gd2.addNumericField("Top:", 5, 0); // this will be enabled only if box is checked

		TextField topNField = (TextField) gd2.getNumericFields().get(6); // get correct index
		topNField.setEnabled(false); // initially off
		gd2.addDialogListener(new DialogListener() {
			public boolean dialogItemChanged(GenericDialog gd2, AWTEvent e) {
				boolean isSpeedEvolutionChecked = ((Checkbox) gd2.getCheckboxes().get(2)).getState();
				topNField.setEnabled(isSpeedEvolutionChecked);
				return true;
			}
		});
		gd2.addCheckbox("Average orientation distribution", false);
		gd2.addMessage("");

		// advanced options
		gd2.addMessage("=== Advanced Options ===");
		gd2.addCheckbox("Display all", false);
		gd2.addChoice("Detection of local max", displayOptions, displayOptions[0]);
		gd2.addToSameRow();
		gd2.addChoice("Detection of spots", displayOptions, displayOptions[0]);
		gd2.addChoice("Costs of spots", displayOptions, displayOptions[0]);

		gd2.showDialog();
		if (gd2.wasCanceled()) return;

		// retrieve the values from second GUI
		//segmentation
		sigma = gd2.getNextNumber();
		threshold = gd2.getNextNumber();
		boolean preview = gd2.getNextBoolean(); // even if not used afterwards, needed for a correct GUI implementation

		//tracking
		costmax = gd2.getNextNumber();
		lambda = gd2.getNextNumber();
		gamma = gd2.getNextNumber();
		kappa = gd2.getNextNumber();

		// trajectories determination and display
		String choiceCostFunc = gd2.getNextChoice();
		String choiceColoring = gd2.getNextChoice();
		String choiceLegend = gd2.getNextChoice();

		// speeds
		boolean choiceAvgSpeedDistrib = gd2.getNextBoolean();
		boolean choiceTopNSpeeds = gd2.getNextBoolean();
		boolean choiceAvgOrientDistrib = gd2.getNextBoolean();
		int topN = (int) gd2.getNextNumber();

		// advanced options
		boolean choiceDisplayAll = gd2.getNextBoolean();
		String choiceDetectLocalMax = gd2.getNextChoice();
		String choiceSpotsDetection = gd2.getNextChoice();
		String choiceSpotsCosts= gd2.getNextChoice();

		// initializing parameters so that all intermediate results are displayed
		// following the choice of the user
		boolean userChoiceLocalMax = false;
		boolean userChoiceCosts = false;
		if(choiceDisplayAll){
			userChoiceLocalMax = true;
			userChoiceCosts = true;
		}

		// display the local maxima detection depending on the choice of the user
		switch (choiceDetectLocalMax) {
			case "Do not display":
				break;
			case "Display":
				userChoiceLocalMax = true;
				break;
		}

		// detection of spots
		PartitionedGraph framesDiff = detect(tempDiff,sigma,threshold, userChoiceLocalMax);

		// display the detected spots if the user has checked the "Display all" checkbox
		if(choiceDisplayAll){
			framesDiff.drawSpots(tempDiff);
		}

		// display the detected spots depending on the choice of the user
		switch (choiceSpotsDetection) {
			case "Do not display":
				break;
			case "Display":
				framesDiff.drawSpots(tempDiff);
				break;
		}

		// set parameters to display costs of spots depending on the choice of the user
		int dimension = 20;
		switch (choiceSpotsCosts) {
			case "Do not display":
				break;
			case "Display":
				userChoiceCosts = true;
				break;
		}

		// track spots across frames using a tailored cost function from two different options
		// depending on the user choice
		AbstractDirCost cost = new DirectionCost(outputstack, costmax, lambda, gamma, kappa);
		PartitionedGraph trajectoriesDiff = new PartitionedGraph();
		switch (choiceCostFunc) {
			case "Balanced with distance, direction and intensity":
				trajectoriesDiff = directionalTracking(framesDiff, cost, dimension, userChoiceCosts, false);
				break;
			case "Balanced with distance, direction, intensity and speed":
				trajectoriesDiff = directionalTracking(framesDiff, cost, dimension, userChoiceCosts, true);
				break;
		}

		// get rid of too short trajectories
		PartitionedGraph cleanTraj = cleaningTrajectories(trajectoriesDiff, 5);

		// get important parameters for speed computation
		Calibration cal = imp.getCalibration();
		double pixelWidth = cal.pixelWidth; // pixel width = height (pixels are considered as square)
		String unit = cal.getUnit();
		double frameInterval = cal.frameInterval; // seconds per frame

		// display one coloring for trajectories from four different options
		// depending on the user choice
		ImagePlus final_imp = tempDiff.duplicate();
		switch (choiceColoring) {
			case "Random":
				final_imp.setTitle("with Random Coloring");
				break;
			case "Global average orientation":
				final_imp.setTitle("with Coloring According to Global Average Orientation");
				colorOrientation(cleanTraj);
				break;
			case "Average local orientation":
				final_imp.setTitle("with Coloring According to Average Local Orientation");
				colorOrientationAverage(cleanTraj);
				break;
			case "Instantaneous velocity":
				final_imp.setTitle("with Coloring According to Instantaneous Velocities");
				colorSpeed(cleanTraj, frameInterval, pixelWidth);
				break;
		}

		// display a color map legend depending on the user choice
		switch (choiceLegend) {
			case "Do not display":
				cleanTraj.drawLines(final_imp, false);
				final_imp.setDisplayRange(0, final_imp.getStatistics().max );
				final_imp.updateAndDraw();
				break;
			case "Legend for orientation coloring":
				cleanTraj.drawLines(final_imp,false);
				final_imp.setDisplayRange(0, final_imp.getStatistics().max );
				final_imp.updateAndDraw();
				addLegend("Orientation track map (rad)", false);
				break;
			case "Legend for velocity coloring":
				cleanTraj.drawLines(final_imp, true);
				final_imp.setDisplayRange(0, final_imp.getStatistics().max );
				final_imp.updateAndDraw();
				addLegend("Velocity track map (um/s)", true);
				break;
		}

		// initializing the number of bins to plot the histograms
		int nbins = Math.round(cleanTraj.size()/4);

		// display average speed distribution of spots over the entire stack
		// depending on the user choice
		 if(choiceAvgSpeedDistrib){
			 double[] speeds = computeAvgSpeed(cleanTraj, frameInterval, pixelWidth);
			 histogram(speeds, nbins, "Average Speed Distribution of Microtubules", "Speed in "+unit+"/s", false);
		 }

		// display the speeds of the topN longest trajectories in a line plot depending
		// on the user choice
		if(choiceTopNSpeeds){
			TreeMap<Integer, TreeMap<Integer, Double>> topNSpeeds = computeTopNSpeed(cleanTraj, frameInterval, topN, pixelWidth);
			linePlot(topNSpeeds, topN, "Speed in "+unit+"/s");
		}

		// display the average orientation of trajectories distribution over the entire stack
		// depending on the user choice
		if(choiceAvgOrientDistrib){
			double[] orientations = computeAvgOrientation(cleanTraj);
			histogram(orientations, nbins, "Average Orientation Distribution of Microtubules", "Orientation in radians", true);
		}

	}


	/**
	 * Functional interface representing a transformation on an ImageProcessor.
	 */
	@FunctionalInterface
	interface ImageProcessorFunction {
		// apply a transformation to an ImageProcessor and returns the result
		ImageProcessor apply(ImageProcessor ip);
	}


	/**
	 * This method allows to process and apply the given function to a whole ImagePlus object.
	 *
	 * @param imp The ImagePlus on which we want to apply the function.
	 * @param func The function that we want to apply to each frame's processor.
	 * @return An ImagePlus object where func has been applied to each image processor.
	 */
	private ImagePlus processStack(ImagePlus imp, ImageProcessorFunction func) {
		ImageStack newStack = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int i = 1; i <= imp.getStackSize(); i++) {
			ImageProcessor ip = imp.getStack().getProcessor(i); // get the ImageProcessor for the current slice
			ImageProcessor result = func.apply(ip);  // call passed function
			newStack.addSlice(result);
			System.gc();  // encourage cleanup
		}

		ImagePlus resultImp = new ImagePlus("Processed", newStack);
		resultImp.setDimensions(1, 1, imp.getStackSize());
		resultImp.setOpenAsHyperStack(true); // ensure the stack is treated as a hyperstack

		return resultImp;
	}


	/**
	 * This method creates a DoG filter processor for a given processor.
	 * The level of blur of the filter is adjusted by the parameter sigma.
	 *
	 * @param ip The ImageProcessor of an ImagePlus object.
	 * @param sigma The level of blur of the gaussian filters.
	 * @return A DoG processor.
	 */
	private ImageProcessor classic_dog(ImageProcessor ip, double sigma) {
		// duplicate the input image twice for two different levels of Gaussian blur
		ImagePlus g1 = new ImagePlus("g1", ip.duplicate());
		ImagePlus g2 = new ImagePlus("g2", ip.duplicate());

		double sigma2 = (Math.sqrt(2) * sigma); // compute sigma for the second Gaussian
		GaussianBlur3D.blur(g1, sigma, sigma, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g1, g2, "Subtract create stack"); // performing DoG

		return dog.getProcessor();
	}


	/**
	 * This method applies a DoG filter on every time frame of the ImagePlus input.
	 * Sigma1 and Sigma2 are unpaired to have better control on the filtering, which
	 * is best suited for our application here.
	 *
	 * @param ip The processor of a given ImagePlus object.
	 * @param sigma1 The level of blur of the first gaussian filter = what objects we want to remove (background).
	 * @param sigma2 The level of blur of the second gaussian filter = what objects we want to keep.
	 * @return ImageProcessor of the DoG filtered image.
	 */
	private ImageProcessor dog(ImageProcessor ip, double sigma1, double sigma2 ) {
		// duplicate the input image twice for two different levels of Gaussian blur
		ImagePlus g1 = new ImagePlus("g1", ip.duplicate());
		ImagePlus g2 = new ImagePlus("g2", ip.duplicate());

		// here sigma 2 is given as parameter and not computed from sigma 1
		GaussianBlur3D.blur(g1, sigma1, sigma1, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g2, g1, "Subtract create stack"); // performing DoG

		return dog.getProcessor();
	}


	/**
	 * This method normalises the image's pixels values for a single processor.
	 *
	 * @param ip The image processor with pixel values to normalise.
	 * @return ImageProcessor with pixel values normalised.
	 */
	private ImageProcessor normalisation(ImageProcessor ip){
		ImagePlus frame = new ImagePlus("f",ip.duplicate());

		ImageStatistics statistics = frame.getStatistics();
		double std = statistics.stdDev;
		double mean = statistics.mean;
		if (std == 0) std = 1; // avoid division by zero

		// normalize pixel intensity values
		IJ.run(frame, "Subtract...", "value="+mean+" slice");
		IJ.run(frame, "Divide...", "value="+std+" slice");

		return frame.getProcessor();
	}


	/**
	 * This method allows to detect bright spots in an image based on a given intensity threshold.
	 * The algorithm is working by going through each pixel of the DoG filtered image, get the corresponding pixel
	 * value from the unfiltered image and compare it to a given threshold. If the pixel value is below the threshold,
	 * it examines the next pixel. If the pixel value is above the threshold, it retrieves the pixel value from the DoG
	 * filtered image and compare it to the values of its surrounding pixels. If it is a local maxima, it stores the
	 * coordinates, the frame number and the corresponding pixel value from the unfiltered image in a list.
	 *
	 * @param dog ImageProcessor containing the DoG filtered image.
	 * @param image Original ImagePlus unfiltered used to apply the intensity threshold.
	 * @param t The current time frame.
	 * @param threshold The minimum intensity in the original image to detect a local maximum.
	 * @return Spots containing all local maxima spots of the current time frame.
	 */
	public Spots localMax(ImageProcessor dog, ImageProcessor image, int t, double threshold) {
		Spots spots = new Spots();

		// going through the image pixel by pixel
		for (int x = 1; x < dog.getWidth() - 1; x++) {
			for (int y = 1; y < dog.getHeight() - 1; y++) {
				double valueImage = image.getPixelValue(x, y);

				// compare value of the current pixel to the threshold
				if (valueImage >= threshold) {
					// if above the threshold we get the corresponding pixel value after applying the DoG filter
					double v = dog.getPixelValue(x, y);
					double max = -1;

					// check neighboring pixels of the detected pixel above the threshold
					for (int k = -1; k <= 1; k++) // k=-1,0,1
						for (int l = -1; l <= 1; l++) // l=-1,0,1
							// save the maximum value between the 9 pixels centered around the current pixel above the threshold
							max = Math.max(max, dog.getPixelValue(x + k, y + l));

					// if the pixel in the center is the max between the 9 pixels, then we add it as a Spot to the list
					if (v == max) spots.add(new Spot(x, y, t, valueImage));
				}
			}
		}

		return spots;
	}


	/**
	 * This method applies a DoG filter on every time frame of the ImagePlus input.
	 * Then every spot on each frame is detected if the intensity is above the specified threshold.
	 * The size of the spots detected is saved into the IJ log and they are added into a partitioned graph
	 * which is returned by the method.
	 *
	 * @param imp The ImagePlus object input.
	 * @param sigma The level of blur of the gaussian filter of the DoG.
	 * @param threshold The threshold of intensity to detect spots in the image input imp.
	 * @return Graph that contains the spots detected.
	 */
	private PartitionedGraph detect(ImagePlus imp, double sigma, double threshold, boolean user_choice) {
		int nt = imp.getNFrames();
		PartitionedGraph graph = new PartitionedGraph();

		for (int t = 0; t < nt; t++) { // loop through all frames
			imp.setPosition(1, 1, 1+t);
			ImageProcessor ip = imp.getProcessor();
			ImageProcessor dog = classic_dog(ip, sigma); // apply classic dog to the given ImagePlus
			Spots spots = localMax(dog, ip, t, threshold); // detect local maxima in DoG image that exceed the threshold
			graph.add(spots); // add the detected spots to the graph

			// display the number of local max on each stack frame
			if (user_choice) {
				IJ.log("Frame t:" + t + " #localmax:" + spots.size() );
			}
		}
		return graph;
	}


	/**
	 * This method performs directional tracking to build trajectories based on a custom directional cost function.
	 * It iterates through each spot in the input, attempting to build forward trajectories by linking spots across
	 * successive time points. For each starting spot, it searches in subsequent frames for the best matching spot based
	 * on the provided cost function, which includes both a cost evaluation and a validation criterion.
	 * A new trajectory is created for each unassigned spot, and it is extended frame-by-frame until no valid
	 * match is found or the end of the sequence is reached. Matching is constrained by a maximum cost threshold
	 * and a spatial dimension constraint.
	 *
	 * @param frames The input Partitioned Graph.
	 * @param cost The implementation of cost and validation criteria to link spots together.
	 * @param dimension The integer specifying a dimensional constraint for cost evaluation.
	 * @return Partitioned Graph where each partition corresponds to a tracked trajectory.
	 */
	private PartitionedGraph directionalTracking(PartitionedGraph frames, AbstractDirCost cost, int dimension, boolean user_choice, boolean speed) {
		PartitionedGraph trajectories = new PartitionedGraph();

		for (Spots frame : frames) { // iterate over all frames
			for (Spot spot : frame) { // iterate over all spots in the current frame
				Spots trajectory = trajectories.getPartitionOf(spot); // get the trajectory this spot is already part of

				// if it's not part of any trajectory yet, create a new one
				if (trajectory == null) trajectory = trajectories.createPartition(spot);

				if (spot.equals(trajectory.last())) { // start tracking only from the last spot of a trajectory
					int t0 = spot.t;

					for (int t=t0; t < frames.size() - 1; t++) { // extend the trajectory to future frames
						double trajectory_cost = this.costmax; // set the first cost value to be the highest possible
						Spot next_spot = null; // // store best match for the next frame (initialized to null)

						for(Spot next : frames.get(t+1)) { // iterate over all spots of the next frame
							if(speed) {
								// if speed constraint is used, validate and evaluate methods taking speed into account
								if (cost.validate_withSpeed(next, spot, frames, dimension) == true) { // if the cost is lesser than the costmax
									// if the new cost is less than the previous one, we save the spot
									trajectory_cost = cost.evaluate_withSpeed(next, spot, frames, dimension);
									next_spot = next;
								}
							}else{
								// otherwise, use cost validate and evaluate methods not taking speed into account
								if (cost.validate(next, spot, frames, dimension) == true) { // if the cost is lesser than the costmax
									// if the new cost is less than the previous one, we save the spot
									trajectory_cost = cost.evaluate(next, spot, frames, dimension);
									next_spot = next;
								}
							}
						}

						if (next_spot != null) { // check that we found a next spot to add to the trajectory
							trajectory.add(next_spot); // final spot is saved in next spot
							spot = next_spot;
							if(user_choice){ // display the chosen spot and its associated cost
								IJ.log("#" + trajectories.size() + " spot " + next_spot + " with a cost:" + trajectory_cost);
							}
						} else {
							break;  // no valid match found, stop this trajectory
						}
					}
				}
			}
		}

		return trajectories;
	}


	/**
	 * This method filters out short trajectories from a Partitioned Graph based on a minimum length threshold.
	 * It iterates through each trajectory in the input graph and retains only those whose length (i.e., number of spots)
	 * exceeds the specified threshold.
	 *
	 * @param frames The input Partitioned Graph.
	 * @param min_length The minimum number of spots a trajectory must have to be retained (threshold).
	 * @return Partitioned Graph containing trajectories only longer than the chosen threshold.
	 */
	private PartitionedGraph cleaningTrajectories(PartitionedGraph frames, int min_length){
		PartitionedGraph final_graph = new PartitionedGraph();

		// clean up minimal trajectories to lighten memory load
		for (Spots trajectory : frames){
			if (trajectory.size() > min_length){
				final_graph.add(trajectory);
			}
		}

		return final_graph;
	}


	/**
	 * This method computes the orientation angle (in radians) of a vector given its x and y components.
	 * The angle is measured counterclockwise from the positive x-axis to the vector, and it is computed using the
	 * built-in Math.atan2() function, which uses cartesian coordinates and correctly handles the sign of the vector.
	 *
	 * @param dx The horizontal component of the vector.
	 * @param dy The vertical component of the vector.
	 * @return The angle in radians in the range [-π, π].
	 */
	public double getOrientation(double dx, double dy){
		return Math.atan2(dy, dx); // angle in radians
	}


	/**
	 * This method computes the angle (in radians) between two spots. The orientation is defined as the angle of the
	 * vector with respect to the horizontal x-axis. Here, this angle can be used to determine the direction of motion
	 * between two points in a trajectory.
	 *
	 * @param start The starting point.
	 * @param end The ending spot.
	 * @return The orientation angle in radians, in the range [-π, π].
	 */
	public double getOrientation(Spot start, Spot end){
		double dx = end.x - start.x;
		double dy = end.y - start.y;

		return getOrientation(dx,dy); // compute orientation angle of the vector between the two given spots
	}


	/**
	 * This method maps the orientation of a vector to a color gradient
	 *
	 * @param orientation Angle of a vector, in radians.
	 * @return Color object being the new color corresponding to the orientation.
	 */
	private Color mapColor(double orientation){
		// convert angle from [-π, π] to a normalized hue value [0, 1]
		float hue = (float) ((orientation + Math.PI) / (2 * Math.PI));
		// create a fully saturated and bright color from the hue (with HSB color model)
		Color color = Color.getHSBColor(hue, 1f, 1f);
		color = new Color(color.getRed(), color.getGreen(), color.getBlue(), 120); // make color semi-transparent

		return color;
	}


	/**
	 * This method assigns a color to each trajectory based on its orientation. For each trajectory in the input graph,
	 * this method computes the "global" orientation by taking the vector from the first to the last spot in the
	 * trajectory. This orientation is then mapped to a specific color, and the trajectory is annotated with this color.
	 *
	 * @param input The input Partitioned Graph containing trajectories.
	 */
	private void colorOrientation(PartitionedGraph input){
		for(Spots trajectory : input) { // loop through all the trajectories

			Spot first_spot = trajectory.first();
			Spot last_spot = trajectory.last();

			// compute the orientation angle from first to last spot
			double orientation = getOrientation(first_spot, last_spot);
			Color newColor = mapColor(orientation); // map the orientation angle to a specific color
			trajectory.color = newColor; // assign the computed color to the entire trajectory
		}
	}


	/**
	 * This method assigns a color to each trajectory based on its average local orientation.
	 * The orientation between each pair of consecutive spots is computed,
	 * and the mean of these orientations is used to determine the color via a colormap.
	 * This method is more robust than using only the global direction, as it accounts for curvature and small changes
	 * in direction across the trajectory.
	 *
	 * @param input The input PartitionedGraph containing trajectories.
	 */
	private void colorOrientationAverage(PartitionedGraph input) {
		for (Spots trajectory : input) { // loop through all trajectories
			if (trajectory.size() < 2) continue; // make sure no division by zero happens

			double sumOrientation = 0;
			int count = 0;

			// compute orientation between all consecutive spot pairs
			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);
				sumOrientation += getOrientation(a, b);
				count++;
			}

			double avgOrientation = sumOrientation / count; // compute average orientation across the trajectory
			Color newColor = mapColor(avgOrientation); // map the average orientation to a color
			trajectory.color = newColor; // assign the color to the entire trajectory
		}
	}


	/**
	 * This method computes the average orientation of trajectories within a graph.
	 *
	 * @param frames Graph with all trajectories.
	 * @return Array of average orientations for each trajectory.
	 */
	private double[] computeAvgOrientation(PartitionedGraph frames) {
		double[] all_orientations = new double[frames.size()];
		int j = 0;

		for (Spots trajectory : frames) {
			if (trajectory.size() < 2) continue; // make sure no division by zero happens

			double sumOrientation = 0;

			// compute orientation angles between all consecutive pairs of Spots
			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);
				sumOrientation += getOrientation(a, b);
			}
			// compute and store average orientation angle
			all_orientations[j] = sumOrientation / (trajectory.size() - 1);
			++j;
		}

		return all_orientations;
	}


	/**
	 * This method maps the value of the speed to a color gradient (red to blue).
	 *
	 * @param speed Value of the speed to map.
	 * @return Color corresponding to the given speed value.
	 */
	private Color mapSpeedColor(double speed){
		double maxSpeed = 4; // upper bound for speed scaling
		// normalize speed and invert hue: 0 → blue (0.66), maxSpeed → red (0.0)
		float hue = 0.66f - 0.66f * Math.min((float)speed / (float)maxSpeed, 1.0f);
		// create a fully saturated and bright color from the hue (with HSB color model)
		Color color = Color.getHSBColor(hue, 1f, 1f);
		color = new Color(color.getRed(), color.getGreen(), color.getBlue(), 120); // make color semi-transparent

		return color;
	}


	/**
	 * This method colors the spots in the trajectories based on their speed.
	 *
	 * @param input Graph input with all trajectories.
	 * @param frameInterval The time interval between two frames (to compute speed).
	 * @param pixelToUm The conversion factor from pixel to micrometers.
	 */
	private void colorSpeed(PartitionedGraph input, double frameInterval, double pixelToUm){
		for(Spots trajectory : input) { // loop over each trajectory in the graph
			trajectory.initSpeedColor();

			// compute the speed between each pair of consecutive spots
			double[] speeds = computeTrajectorySpeeds(trajectory, frameInterval, pixelToUm);

			// map each speed to a color and store it
			for (int i=0; i < speeds.length; ++i){
				Color speedColor = mapSpeedColor(speeds[i]);
				trajectory.speed_color[i] = speedColor;
			}
		}
	}


	/**
	 * This method computes the average speed of a trajectory for all frames within a partitioned graph.
	 *
	 * @param frames Graph that contains all the trajectories.
	 * @param frameInterval Time interval between two frames.
	 * @param pixelToUm Conversion factor for pixel to um.
	 * @return All the average speeds for all the trajectories in a list.
	 */
	private double[] computeAvgSpeed(PartitionedGraph frames, double frameInterval, double pixelToUm){
		double[] all_speeds = new double[frames.size()];
		int j = 0;

		for(Spots trajectory : frames){
			if (trajectory.size() < 2) continue; // make sure no division by zero happens

			double avg_speed = 0.0;

			// compute sum of speeds between each pair of consecutive Spot
			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);

				double speedAtoB = b.distance(a) / frameInterval; // speed from a to b
				avg_speed += speedAtoB;
			}
			// normalize by the number of segments and convert to micrometers
			avg_speed = avg_speed / (trajectory.size()-1);
			all_speeds[j] = avg_speed*pixelToUm;
			++j;
		}

		return all_speeds;
	}


	/**
	 * This method computes all the instantaneous speeds for each point of a given trajectory. First speed is set to zero.
	 *
	 * @param trajectory The trajectory we want to compute the speeds for.
	 * @param frameInterval The time interval between two frames.
	 * @param pixelToUm The conversion factor from pixel to um.
	 * @return An array that contains all the speeds in the trajectory, of same size as the number of spots in the trajectory.
	 */
	private double[] computeTrajectorySpeeds(Spots trajectory, double frameInterval, double pixelToUm){
		double[] trajectory_all_speeds = new double[trajectory.size()];
		trajectory_all_speeds[0] = 0; // first element has no previous point to compare, speed is set to 0

		// loop through trajectory and compute speed between each pair of Spot
		for (int i = 0; i < trajectory.size() - 1; i++) {
			Spot a = trajectory.get(i);
			Spot b = trajectory.get(i + 1);
			
			double speedAtoB = b.distance(a) / frameInterval; // speed from a to b
			trajectory_all_speeds[i+1] = speedAtoB*pixelToUm; // convert speed to micrometers per time unit and store it
		}

		return trajectory_all_speeds;
	}


	/**
	 * This method computes the instantaneous speeds for the top N trajectories.
	 *
	 * @param frames Graph containing all the trajectories.
	 * @param frameInterval Time interval between two frames.
	 * @param topN The number of longest trajectories we want to compute.
	 * @return All the instantaneous speeds for the top N trajectories.
	 */
	private TreeMap<Integer, TreeMap<Integer, Double>> computeTopNSpeed(PartitionedGraph frames, double frameInterval, int topN, double pixelToUm){
		TreeMap<Integer, TreeMap<Integer, Double>> speedMap = new TreeMap<>();
		PartitionedGraph trajToCompute = new PartitionedGraph();

		// sort trajectories by size in ascending order
		frames.sort(Comparator.comparingInt(traj -> traj.size()));
		// select the top-N longest trajectories (from the end of the sorted list)
		for(int i = frames.size() - 1; i > frames.size() - 1 - topN; --i){
			trajToCompute.add(frames.get(i));
		}

		int trajId = 0;
		// store speed values starting from second spot (first speed is for transition from 0 to 1)
		for(Spots trajectory : trajToCompute){
			TreeMap<Integer, Double> trajSpeeds = new TreeMap<>();
			double[] speeds = computeTrajectorySpeeds(trajectory, frameInterval, pixelToUm);

			for (int i = 1; i < trajectory.size(); i++) {
				Spot b = trajectory.get(i);
				trajSpeeds.put(b.t, speeds[i]); // speed from previous spot to b
			}
			++trajId;
			speedMap.put(trajId, trajSpeeds); // store trajectory's speed map by ID
		}

		return speedMap;
	}


	/**
	 * This method adds a horizontal color map legend below each frame of an ImagePlus stack and displays the modified
	 * image. The legend is a color bar representing orientations of trajectories from -π to π.
	 * This method pads each frame of the original image vertically to make room for the legend, copies the original
	 * content into the new frame, fills the padded region with white, and then draws a color gradient representing
	 * angles from -π to π using a color mapping function.
	 *
	 * @param legend_title Title for the color map legend.
	 */
	private void addLegend(String legend_title, boolean withSpeed) {
		int legendWidth = 500;
		int legendHeight = 50;
		int labelHeight = 60; // extra height for labels and title
		int totalHeight = legendHeight + labelHeight;

		// create a blank color image
		ImageProcessor legendIp = new ColorProcessor(legendWidth, totalHeight);
		legendIp.setColor(Color.WHITE); // fill background with white
		legendIp.fill();

		if(withSpeed){
			// draw color bar for speed
			double maxSpeed = 5.0 ; // TODO: find a way to have maxSpeed robust and not hardcoded
			for (int i = 0; i < legendWidth; i++) {
				double speed = maxSpeed * i / (double) (legendWidth - 1);
				Color c = mapSpeedColor(speed);
				for (int j = 0; j < legendHeight; j++) {
					legendIp.setColor(c);
					legendIp.drawPixel(i, j);
				}
			}
			// draw labels for speed
			legendIp.setColor(Color.BLACK);
			legendIp.setFont(new Font("SansSerif", Font.PLAIN, 14));
			legendIp.drawString("0", 0, legendHeight + 20);
			legendIp.drawString(String.format("%.1f", maxSpeed), legendWidth - 40, legendHeight + 20);

		}else{
			// draw the color bar
			for (int i = 0; i < legendWidth; i++) {
				double angle = (2 * Math.PI) * i / (double) (legendWidth - 1);
				Color c = mapColor(angle);
				for (int j = 0; j < legendHeight; j++) {
					legendIp.setColor(c);
					legendIp.drawPixel(i, j);
				}
			}
			// draw labels for orientation
			legendIp.setColor(Color.BLACK);
			legendIp.setFont(new Font("SansSerif", Font.PLAIN, 14));
			legendIp.drawString("-π", 0, legendHeight + 20);
			legendIp.drawString("π", legendWidth - 20, legendHeight + 20);

		}

		// title centered
		legendIp.setFont(new Font("SansSerif", Font.PLAIN, 18));
		int titleX = (legendWidth - legendIp.getStringWidth(legend_title)) / 2;
		legendIp.drawString(legend_title, titleX, legendHeight + 45);

		// show in new window
		ImagePlus legendImp = new ImagePlus("Legend", legendIp);
		legendImp.show();
	}


	/**
	 * This method plots a histogram.
	 *
	 * @param toplot Array of data to plot.
	 * @param nbins Number of bins of the histogram.
	 * @param title Title of the plot.
	 * @param xlabel Label on x axis.
	 * @param setXForAngle Conversion factor to change x units to radians.
	 */
	private void histogram(double[] toplot, int nbins, String title, String xlabel, boolean setXForAngle){
		double min = Arrays.stream(toplot).min().orElse(0);
		double max = Arrays.stream(toplot).max().orElse(1);
		double binWidth = (max - min) / nbins;

		// initialize bins
		double[] binCenters = new double[nbins];
		double[] frequencies = new double[nbins];
		for (int i = 0; i < nbins; i++) {
			binCenters[i] = min + binWidth * (i + 0.5);
		}

		// fill frequencies
		for (double value : toplot) {
			int bin = (int) ((value - min) / binWidth);
			if (bin >= nbins) bin = nbins - 1;  // Handle max edge
			frequencies[bin]++;
		}

		// create and show plot
		Plot plot = new Plot(title, xlabel, "Frequency");
		if(setXForAngle){ // if plotting a histogram for orientation angles
			plot.setLimits(-Math.PI, Math.PI, 0, Arrays.stream(frequencies).max().orElse(1));
			plot.setColor(Color.RED);
		}else{ // if plotting a histogram for speeds
			plot.setLimits(min, max, 0, Arrays.stream(frequencies).max().orElse(1));
			plot.setColor(Color.BLUE);
		}
		plot.addPoints(binCenters, frequencies, Plot.BAR);
		plot.show();
	}


	/**
	 * This method creates a line plot of the trajectories speed in time.
	 *
	 * @param toplot Data to plot.
	 * @param topN Number of trajectories we plot.
	 * @param ylabel Label of the y-axis.
	 */
	private void linePlot(TreeMap<Integer, TreeMap<Integer, Double>> toplot, int topN, String ylabel){
		Plot plot = new Plot("Speeds from the Top " + topN + " Longest Trajectories", "Frames", ylabel);

		// loop through each trajectory's data
		for (Map.Entry<Integer, TreeMap<Integer, Double>> entry : toplot.entrySet()) {
			TreeMap<Integer, Double> trajSpeeds = entry.getValue();

			// convert TreeMap keys and values to arrays for plotting
			double[] timePoints = trajSpeeds.keySet().stream().mapToDouble(t -> t).toArray();
			double[] speeds = trajSpeeds.values().stream().mapToDouble(s -> s).toArray();

			// assign a unique hue to each trajectory based on its ID
			plot.setColor(Color.getHSBColor((float) entry.getKey() / toplot.size(), 1f, 1f));
			plot.addPoints(timePoints, speeds, Plot.LINE); // add the trajectory's points to the plot as a line
		}
		plot.show();
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
