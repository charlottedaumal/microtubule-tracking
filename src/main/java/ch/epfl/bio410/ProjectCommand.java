package ch.epfl.bio410;

import ch.epfl.bio410.cost.AbstractDirCost;
import ch.epfl.bio410.cost.DirectionCost;
import ch.epfl.bio410.cost.SimpleDistanceCost;
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
	private double sigma ;  // sigma of the DoG
	private double sigma1 ; // sigma1 of preprocessing : to remove background
	private double sigma2 ; // sigma2 of preprocessing : what we want to keep from image
	private int windowDiff =1; // size of window to delete continuous tracks
	private double threshold ; // threshold of intensity to detect spots
	private double lambda ; // weight of distance in cost computation
	private double gamma ; // weight of direction in cost computation
	private double kappa ;

	public void run() {

		// prompt the user to select a file
		OpenDialog fileChooser = new OpenDialog("Select a file", null);
		String filePath = fileChooser.getDirectory();
		String fileName = fileChooser.getFileName();
		
		ImagePlus imp = IJ.openImage(filePath+fileName);

		GenericDialog gd = new GenericDialog("Welcome to the MiTrack Plugin :)");
		//TODO change message text in bold if possible ??

		// === Parameter Tuning Part I Section ===
		gd.addMessage("=== Parameter Tuning Part I ===");

		String[] displayOptions = {
				"Do not display",
				"Display",
		};

		// Preprocessing
		gd.addMessage("Preprocessing");
		gd.addNumericField("Sigma1:", 5,2);
		gd.addToSameRow();
		gd.addNumericField("Sigma2:", 1.25,2);
		gd.addMessage("");

		// === Slider for WindowExp ===

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

		// Wrap slider and label in AWT Panel
		Panel sliderPanel = new Panel(new BorderLayout());
		sliderPanel.add(slider, BorderLayout.CENTER);
		sliderPanel.add(valueLabel, BorderLayout.WEST);
		gd.addPanel(sliderPanel);

		gd.addMessage("");
		gd.addChoice("Preprocessed Stack", displayOptions, displayOptions[0]);

		gd.showDialog();
		if (gd.wasCanceled()) return;

		// retrieve the values from GUI for preprocessing
		sigma1 = gd.getNextNumber();
		sigma2 = gd.getNextNumber();
		int chosenWindowExp = windowExp[0];
		String choicePreprocessed = gd.getNextChoice();

		// convert the image to 32 bits for downstream calculations
		IJ.run(imp, "32-bit", "");
		imp.setTitle("Original Image");
		imp.show();

		ImagePlus result = processStack(imp, ip -> dog(ip, sigma1, sigma2));
		ImagePlus outputstack = processStack(result, this::normalisation);
		outputstack.setTitle("Preprocessed Stack");
		IJ.run(outputstack, "Median...", "radius=1 stack");

		switch (choicePreprocessed) {
			case "Do not display":
				break;
			case "Display":
				outputstack.setDisplayRange(0, outputstack.getStatistics().max);
				outputstack.updateAndDraw();
				outputstack.show();
				break;
		}

		// temporal projection using the wrapper
		TemporalProjector tp = new TemporalProjector(outputstack, "sum", "left", chosenWindowExp);
		ImagePlus temporalExposure = processStack(outputstack, (ImageProcessor ip) -> tp.apply(ip));

		TemporalDifferencer diff = new TemporalDifferencer(temporalExposure, windowDiff);
		ImagePlus tempDiff = processStack(temporalExposure, (ImageProcessor ip) -> diff.apply(ip));
		tempDiff.setTitle("Temporal Projection");
		tempDiff.setDisplayRange(0, tempDiff.getStatistics().max);
		tempDiff.updateAndDraw();
		tempDiff.show();

		NonBlockingGenericDialog gd2 = new NonBlockingGenericDialog("Select further parameters!");

		// === Parameter Tuning Part II Section ===
		gd2.addMessage("=== Parameter Tuning Part II ===");

		// Segmentation
		gd2.addMessage("Segmentation");
		gd2.addNumericField("Sigma:", 1, 2);
		gd2.addToSameRow();
		gd2.addNumericField("Threshold", 5, 3);

		// retrieve the values from GUI
		//segmentation
		sigma = gd2.getNextNumber();
		threshold = gd2.getNextNumber();

		// Add preview checkbox for the threshold
		gd2.addPreviewCheckbox(null);
		gd2.addDialogListener((dialog, e) -> {
			if (gd2.wasCanceled()) return false;
			if (gd2.getPreviewCheckbox().getState()) {
				// We use 2 frames after the inital window size to have a correct print
				int frame = chosenWindowExp+2;
				ImagePlus singleFrame = new ImagePlus("Frame " + (frame),
						tempDiff.getStack().getProcessor(frame).duplicate());
				singleFrame.setDimensions(1, 1, 1);  // Set to a single slice
				PartitionedGraph preview = detect(singleFrame,sigma,threshold, false);
				preview.drawSpots(singleFrame);
			}
			return true;
		});

		// Tracking
		gd2.addMessage("Tracking");
		gd2.addNumericField("Costmax:", 0.5, 3);
		gd2.addToSameRow();
		gd2.addNumericField("Lambda", 0.5, 3);
		gd2.addNumericField("Gamma", 0.3, 3);
		gd2.addToSameRow();
		gd2.addNumericField("Kappa", 0.15, 3);
		gd2.addMessage("");

		// === Results Options ===
		gd2.addMessage("=== Results options ===");

		String[] coloringOptions = {
				"Random",
				"Global average orientation",
				"Average local orientation",
		};
		String[] costOptions = {
				"Balanced with distance, direction and intensity",
				"Balanced with distance, direction, intensity and speed",
		};

		// Trajectories determination and display
		gd2.addChoice("Cost function", costOptions, costOptions[0]);
		gd2.addChoice("Coloring of Trajectories", coloringOptions, coloringOptions[0]);
		gd2.addToSameRow();
		gd2.addChoice("Color map legend", displayOptions, displayOptions[0]);

		// Speeds
		gd2.addCheckbox("Average speed distribution", false);
		gd2.addToSameRow();
		gd2.addCheckbox("Speed evolution from TopN longest trajectories", false);
		gd2.addToSameRow();
		gd2.addNumericField("Top:", 5, 0); // this will be enabled only if box is checked

		TextField topNField = (TextField) gd2.getNumericFields().get(6); // get correct index
		topNField.setEnabled(false); // initially off
		gd2.addDialogListener(new DialogListener() {
			public boolean dialogItemChanged(GenericDialog gd2, AWTEvent e) {
				boolean isSpeedEvolutionChecked = ((Checkbox) gd2.getCheckboxes().get(1)).getState();
				topNField.setEnabled(isSpeedEvolutionChecked);
				return true;
			}
		});
		gd2.addCheckbox("Average orientation distribution", false);
		gd2.addMessage("");

		// === Advanced Options ===
		gd2.addMessage("=== Advanced Options ===");
		gd2.addCheckbox("Display all", false);
		gd2.addChoice("Detection of local max", displayOptions, displayOptions[0]);
		gd2.addToSameRow();
		gd2.addChoice("Detection of spots", displayOptions, displayOptions[0]);
		gd2.addChoice("Costs of spots", displayOptions, displayOptions[0]);





		gd2.showDialog();
		if (gd2.wasCanceled()) return;



		//tracking
		costmax = gd2.getNextNumber();
		lambda = gd2.getNextNumber();
		gamma = gd2.getNextNumber();
		kappa = gd2.getNextNumber();

		String choiceCostFunc = gd2.getNextChoice();
		String choiceColoring = gd2.getNextChoice();
		String choiceLegend = gd2.getNextChoice();
		boolean choiceAvgSpeedDistrib = gd2.getNextBoolean();
		boolean choiceTopNSpeeds = gd2.getNextBoolean();
		boolean choiceAvgOrientDistrib = gd2.getNextBoolean();
		int topN = (int) gd2.getNextNumber();
		boolean choiceDisplayAll = gd2.getNextBoolean();
		String choiceDetectLocalMax = gd2.getNextChoice();
		String choiceSpotsDetection = gd2.getNextChoice();
		String choiceSpotsCosts= gd2.getNextChoice();

		boolean userChoiceLocalMax = false;
		boolean userChoiceCosts = false;

		if(choiceDisplayAll){
			userChoiceLocalMax = true;
			userChoiceCosts = true;
		}

		switch (choiceDetectLocalMax) {
			case "Do not display":
				break;
			case "Display":
				userChoiceLocalMax = true;
				break;
		}

		PartitionedGraph framesDiff = detect(tempDiff,sigma,threshold, userChoiceLocalMax);

		if(choiceDisplayAll){
			framesDiff.drawSpots(tempDiff);
		}

		switch (choiceSpotsDetection) {
			case "Do not display":
				break;
			case "Display":
				framesDiff.drawSpots(tempDiff);
				break;
		}

		int dimension = 20;
		switch (choiceSpotsCosts) {
			case "Do not display":
				break;
			case "Display":
				userChoiceCosts = true;
				break;
		}

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

		PartitionedGraph cleanTraj = cleaningTrajectories(trajectoriesDiff, 5);

		ImagePlus final_imp = tempDiff.duplicate();
		switch (choiceColoring) {
			case "Random":
				final_imp.setTitle("with Random Coloring");
				break;
			case "Global average orientation":
				final_imp.setTitle("with Global Average Orientation Coloring");
				colorOrientation(cleanTraj);
				break;
			case "Average local orientation":
				final_imp.setTitle("with Average Local Orientation Coloring");
				colorOrientationAverage(cleanTraj);
				break;
		}

		switch (choiceLegend) {
			case "Do not display":
				cleanTraj.drawLines(final_imp);
				final_imp.setDisplayRange(0, final_imp.getStatistics().max );
				final_imp.updateAndDraw();
				break;
			case "Display":
				cleanTraj.drawLines(final_imp);
				final_imp.setDisplayRange(0, final_imp.getStatistics().max );
				final_imp.updateAndDraw();
				addLegend("Orientation track map (rad)");
				break;
		}

		Calibration cal = imp.getCalibration();
		double pixelWidth = cal.pixelWidth;
		double pixelHeight = cal.pixelHeight;
		String unit = cal.getUnit();
		// TODO what if the pixelWidth and pixelHeight is not the same ??
		double frameInterval = cal.frameInterval; // seconds per frame

		 if(choiceAvgSpeedDistrib){
			 double[] speeds = computeAvgSpeed(cleanTraj, frameInterval, pixelWidth);
			 histogram(speeds, 10, "Average Speed Distribution of Microtubules", "Speed in "+unit+"/s", false);
		 }
		if(choiceTopNSpeeds){
			TreeMap<Integer, TreeMap<Integer, Double>> topNSpeeds = computeTopNSpeed(cleanTraj, frameInterval, topN);
			pointPlot(topNSpeeds, topN, "Speed in "+unit+"/s");
		}

		if(choiceAvgOrientDistrib){
			double[] orientations = computeAvgOrientation(cleanTraj);
			histogram(orientations, 10, "Average Orientation Distribution of Microtubules", "Orientation in radians", true);
		}
	}


	/**
	 * Functional interface representing a transformation on an ImageProcessor.
	 */
	@FunctionalInterface
	interface ImageProcessorFunction {
		ImageProcessor apply(ImageProcessor ip);
	}


	/**
	 * This method allows to process and apply the given function to a whole ImagePlus object.
	 *
	 * @param imp the ImagePlus on which we want to apply the function
	 * @param func the function that we want to apply to each frame's processor
	 * @return ImagePlus object where func has been applied to each image processor
	 */
	private ImagePlus processStack(ImagePlus imp, ImageProcessorFunction func) {
		ImageStack newStack = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int i = 1; i <= imp.getStackSize(); i++) {
			ImageProcessor ip = imp.getStack().getProcessor(i);
			ImageProcessor result = func.apply(ip);  // call passed function
			newStack.addSlice(result);
			System.gc();  // encourage cleanup
		}

		ImagePlus resultImp = new ImagePlus("Processed", newStack);
		resultImp.setDimensions(1, 1, imp.getStackSize());
		resultImp.setOpenAsHyperStack(true);

		return resultImp;
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
		GaussianBlur3D.blur(g1, sigma1, sigma1, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g2, g1, "Subtract create stack");

		return dog.getProcessor();
	}


	/**
	 * This method normalises the image's pixels values for a single processor.
	 *
	 * @param ip the image processor with pixel values to normalise
	 * @return ImageProcessor with pixel values normalised
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


	/**
	 * This method allows to detect bright spots in an image based on a given intensity threshold.
	 * The algorithm is working by going through each pixel of the DoG filtered image, get the corresponding pixel
	 * value from the unfiltered image and compare it to a given threshold. If the pixel value is below the threshold,
	 * it examines the next pixel. If the pixel value is above the threshold, it retrieves the pixel value from the DoG
	 * filtered image and compare it to the values of its surrounding pixels. If it is a local maxima, it stores the
	 * coordinates, the frame number and the corresponding pixel value from the unfiltered image in a list.
	 *
	 * @param dog ImageProcessor containing the DoG filtered image
	 * @param image original ImagePlus unfiltered used to apply the intensity threshold
	 * @param t the current time frame
	 * @param threshold the minimum intensity in the original image to detect a local maximum
	 * @return Spots containing all local maxima spots of the current time frame.
	 */
	public Spots localMax(ImageProcessor dog, ImageProcessor image, int t, double threshold) {
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

		return spots;
	}


	/**
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
	private PartitionedGraph detect(ImagePlus imp, double sigma, double threshold, boolean user_choice) {

		int nt = imp.getNFrames();
		new ImagePlus("DoG", classic_dog(imp.getProcessor(), sigma));
		PartitionedGraph graph = new PartitionedGraph();
		for (int t = 0; t < nt; t++) {
			imp.setPosition(1, 1, 1+t);
			ImageProcessor ip = imp.getProcessor();
			ImageProcessor dog = classic_dog(ip, sigma);
			Spots spots = localMax(dog, ip, t, threshold);
			graph.add(spots);
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
	 * @param frames input Partitioned Graph
	 * @param cost the implementation of cost and validation criteria to link spots together
	 * @param dimension the integer specifying a dimensional constraint for cost evaluation
	 * @return Partitioned Graph where each partition corresponds to a tracked trajectory
	 */
	private PartitionedGraph directionalTracking(PartitionedGraph frames, AbstractDirCost cost, int dimension, boolean user_choice, boolean speed) {
		PartitionedGraph trajectories = new PartitionedGraph();
		for (Spots frame : frames) {
			for (Spot spot : frame) {
				Spots trajectory = trajectories.getPartitionOf(spot);
				if (trajectory == null) trajectory = trajectories.createPartition(spot);
				if (spot.equals(trajectory.last())) {
					int t0 = spot.t;
					for (int t=t0; t < frames.size() - 1; t++) {
						double trajectory_cost = this.costmax; // set the first cost value to be the highest possible
						Spot next_spot = null; // we first initialize the next_spot to be null
						for(Spot next : frames.get(t+1)) { // iterate over all spots of the next frame
							if (cost.validate(next, spot, frames, dimension) == true) { // if the cost is lesser than the costmax
								if(speed){
									if(cost.evaluate_withSpeed(next, spot, frames, dimension) < trajectory_cost) {
										// if the new cost is less than the previous one we save the spot
										trajectory_cost = cost.evaluate(next, spot, frames, dimension);
										next_spot = next;
									}
								}else{
									if(cost.evaluate(next, spot, frames, dimension) < trajectory_cost) {
										// if the new cost is less than the previous one we save the spot
										trajectory_cost = cost.evaluate(next, spot, frames, dimension);
										next_spot = next;
									}
								}
							}
						}
						if (next_spot != null) { // check that we found a next spot to add to the trajectory
							// after iteration over all spots on next frame, final spot is saved in next spot
							trajectory.add(next_spot);
							spot = next_spot;
							if(user_choice){
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
	 * @param frames input Partitioned Graph
	 * @param min_length the minimum number of spots a trajectory must have to be retained (threshold)
	 * @return Partitioned Graph containing trajectories only longer than the chosen threshold
	 */
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
	 * This method computes the orientation angle (in radians) of a vector given its x and y components.
	 * The angle is measured counterclockwise from the positive x-axis to the vector, and it is computed using the
	 * built-in Math.atan2() function, which uses cartesian coordinates and correctly handles the sign of the vector.
	 *
	 * @param dx the horizontal component of the vector
	 * @param dy the vertical component of the vector
	 * @return the angle in radians in the range [-π, π]
	 */
	public double getOrientation(double dx, double dy){
		return Math.atan2(dy, dx); // in radians
	}


	/**
	 * This method computes the angle (in radians) between two spots. The orientation is defined as the angle of the
	 * vector with respect to the horizontal x-axis. Here, this angle can be used to determine the direction of motion
	 * between two points in a trajectory.
	 *
	 * @param start the starting point
	 * @param end the ending spot
	 * @return the orientation angle in radians, in the range [-π, π]
	 */
	public double getOrientation(Spot start, Spot end){
		double dx = end.x - start.x;
		double dy = end.y - start.y;
		return getOrientation(dx,dy);
	}


	/**
	 * This method maps the orientation of a vector to a color gradient
	 *
	 * @param orientation angle of a vector, in radians
	 * @return Color object being the new color corresponding to the orientation
	 */
	private Color mapColor(double orientation){
		float hue = (float) ((orientation + Math.PI) / (2 * Math.PI));
		Color color = Color.getHSBColor(hue, 1f, 1f);
		color = new Color(color.getRed(), color.getGreen(), color.getBlue(), 120);
		return color;
	}


	/**
	 * This method assigns a color to each trajectory based on its orientation. For each trajectory in the input graph,
	 * this method computes the "global" orientation by taking the vector from the first to the last spot in the
	 * trajectory. This orientation is then mapped to a specific color, and the trajectory is annotated with this color.
	 *
	 * @param input input Partitioned Graph containing trajectories
	 */
	private void colorOrientation(PartitionedGraph input){
		PartitionedGraph out = new PartitionedGraph();
		SimpleDistanceCost dist = new SimpleDistanceCost(this.costmax);

		for(Spots trajectory : input) {// looping through all the trajectories

			Spot first_spot = trajectory.get(0);
			Spot last_spot = trajectory.get(trajectory.size()-1);

			double orientation = getOrientation(first_spot, last_spot);
			Color newColor = mapColor(orientation);
			trajectory.color = newColor;
		}
	}


	/**
	 * This method assigns a color to each trajectory based on its average local orientation.
	 * The orientation between each pair of consecutive spots is computed,
	 * and the mean of these orientations is used to determine the color via a colormap.
	 * This method is more robust than using only the global direction, as it accounts for curvature and small changes
	 * in direction across the trajectory.
	 *
	 * @param input input PartitionedGraph containing trajectories
	 */
	private void colorOrientationAverage(PartitionedGraph input) {
		for (Spots trajectory : input) {
			if (trajectory.size() < 2) continue; // to make sure no division by zero happens

			double sumOrientation = 0;
			int count = 0;

			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);
				sumOrientation += getOrientation(a, b);
				count++;
			}

			double avgOrientation = sumOrientation / count;
			Color newColor = mapColor(avgOrientation);
			trajectory.color = newColor;
		}
	}


	/**
	 * This method adds a horizontal color map legend below each frame of an ImagePlus stack and displays the modified
	 * image. The legend is a color bar representing orientations of trajectories from -π to π.
	 * This method pads each frame of the original image vertically to make room for the legend, copies the original
	 * content into the new frame, fills the padded region with white, and then draws a color gradient representing
	 * angles from -π to π using a color mapping function.
	 *
	 * @param legend_title title for the color map legend
	 */
	private void addLegend(String legend_title) {
		int legendWidth = 500;
		int legendHeight = 50;
		int labelHeight = 60; // extra height for labels and title
		int totalHeight = legendHeight + labelHeight;

		ImageProcessor legendIp = new ColorProcessor(legendWidth, totalHeight);
		legendIp.setColor(Color.WHITE); // Fill background with white
		legendIp.fill();

		// Draw the color bar
		for (int i = 0; i < legendWidth; i++) {
			double angle = (2 * Math.PI) * i / (double) (legendWidth - 1);
			Color c = mapColor(angle);
			for (int j = 0; j < legendHeight; j++) {
				legendIp.setColor(c);
				legendIp.drawPixel(i, j);
			}
		}

		// Draw labels and title
		legendIp.setColor(Color.BLACK);
		legendIp.setFont(new Font("SansSerif", Font.PLAIN, 14));
		legendIp.drawString("-π", 0, legendHeight + 20);
		legendIp.drawString("π", legendWidth - 20, legendHeight + 20);

		// Title centered
		legendIp.setFont(new Font("SansSerif", Font.PLAIN, 18));
		int titleX = (legendWidth - legendIp.getStringWidth(legend_title)) / 2;
		legendIp.drawString(legend_title, titleX, legendHeight + 45);

		// Show in new window
		ImagePlus legendImp = new ImagePlus("Legend", legendIp);
		legendImp.show();
	}

	/**
	 * TODO
	 * @param frames
	 * @param frameInterval
	 * @param pixelToUm
	 * @return
	 */
	private double[] computeAvgSpeed(PartitionedGraph frames, double frameInterval, double pixelToUm){
		double[] all_speeds = new double[frames.size()];
		int j = 0;

		for(Spots trajectory : frames){
			if (trajectory.size() < 2) continue; // to make sure no division by zero happens

			double avg_speed = 0.0;

			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);

				double dx     = b.x - a.x;
				double dy     = b.y - a.y;
				double speedAtoB = Math.sqrt(dx*dx + dy*dy ) / frameInterval; // speed from a to b

				avg_speed += speedAtoB;
			}
			avg_speed = avg_speed / (trajectory.size()-1);
			all_speeds[j] = avg_speed*pixelToUm;
			++j;
		}
		return all_speeds;
	}


	/**
	 * TODO
	 * @param frames
	 * @param frameInterval
	 * @param topN
	 * @return
	 */
	private TreeMap<Integer, TreeMap<Integer, Double>> computeTopNSpeed(PartitionedGraph frames, double frameInterval, int topN){
		TreeMap<Integer, TreeMap<Integer, Double>> speedMap = new TreeMap<>();

		frames.sort(Comparator.comparingInt(traj -> traj.size()));

		PartitionedGraph trajToCompute = new PartitionedGraph();

		for(int i=frames.size()-1; i > frames.size() - 1 -topN; --i){
			trajToCompute.add(frames.get(i));
		}

		int trajId = 0;
		for(Spots trajectory : trajToCompute){
			TreeMap<Integer, Double> trajSpeeds = new TreeMap<>();

			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);

				double dx     = b.x - a.x;
				double dy     = b.y - a.y;
				double speedAtoB = Math.sqrt(dx*dx + dy*dy ) / frameInterval; // speed from a to b

				trajSpeeds.put(b.t, speedAtoB);
			}
			++trajId;
			speedMap.put(trajId, trajSpeeds);
		}
		return speedMap;
	}


	/**
	 * TODO
	 * @param frames
	 * @return
	 */
	private double[] computeAvgOrientation(PartitionedGraph frames) {
		double[] all_orientations = new double[frames.size()];
		int j = 0;

		for (Spots trajectory : frames) {
			if (trajectory.size() < 2) continue; // to make sure no division by zero happens

			double sumOrientation = 0;

			for (int i = 0; i < trajectory.size() - 1; i++) {
				Spot a = trajectory.get(i);
				Spot b = trajectory.get(i + 1);
				sumOrientation += getOrientation(a, b);
			}
			all_orientations[j] = sumOrientation / (trajectory.size() - 1);
			++j;
		}
		return all_orientations;
	}


	/**
	 * TODO
	 * @param toplot
	 * @param nbins
	 * @param title
	 * @param xlabel
	 * @param setXForAngle
	 */
	private void histogram(double[] toplot, int nbins, String title, String xlabel, boolean setXForAngle){
		double min = Arrays.stream(toplot).min().orElse(0);
		double max = Arrays.stream(toplot).max().orElse(1);
		double binWidth = (max - min) / nbins;

		// Initialize bins
		double[] binCenters = new double[nbins];
		double[] frequencies = new double[nbins];
		for (int i = 0; i < nbins; i++) {
			binCenters[i] = min + binWidth * (i + 0.5);
		}

		// Fill frequencies
		for (double value : toplot) {
			int bin = (int) ((value - min) / binWidth);
			if (bin >= nbins) bin = nbins - 1;  // Handle max edge
			frequencies[bin]++;
		}

		// Create and show plot
		Plot plot = new Plot(title, xlabel, "Frequency");
		if(setXForAngle){
			plot.setLimits(-Math.PI, Math.PI, 0, Arrays.stream(frequencies).max().orElse(1));
			plot.setColor(Color.RED);
		}else{
			plot.setLimits(min, max, 0, Arrays.stream(frequencies).max().orElse(1));
			plot.setColor(Color.BLUE);
		}
		plot.addPoints(binCenters, frequencies, Plot.BAR);
		plot.show();
	}


	/**
	 * TODO
	 * @param toplot
	 * @param topN
	 * @param ylabel
	 */
	private void pointPlot(TreeMap<Integer, TreeMap<Integer, Double>> toplot, int topN, String ylabel){
		Plot plot = new Plot("Speeds from the Top " + topN + " Longest Trajectories", "Frames", ylabel);

		for (Map.Entry<Integer, TreeMap<Integer, Double>> entry : toplot.entrySet()) {
			TreeMap<Integer, Double> trajSpeeds = entry.getValue();

			double[] timePoints = trajSpeeds.keySet().stream().mapToDouble(t -> t).toArray();
			double[] speeds = trajSpeeds.values().stream().mapToDouble(s -> s).toArray();

			plot.setColor(Color.getHSBColor((float) entry.getKey() / toplot.size(), 1f, 1f)); // optional: color per trajectory
			plot.addPoints(timePoints, speeds, Plot.LINE); // Plot.LINE connects points
		}
		plot.show();
	}


	// Below are functions we coded, but we don't use anymore
	/* TODO: decide whether we should keep them or not (for me we can delete them now, they are still accessible in the
	    history of Gitlab)
	 */


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
	 * This method computes the desired intensity projection on the t axis on the ImagePlus input
	 * to increase contrast and exposure time of the original image. It is done so by a sliding window
	 * that takes the total amount of frames 2*window + 1 centered around the considered frame and sums them.
	 *
	 * @param imp ImagePlus input
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

		ImagePlus resultImp = new ImagePlus(title, results);
		resultImp.setDimensions(1, 1, nFrames);
		resultImp.setOpenAsHyperStack(true);

		return  resultImp;
	}


	/**
	 * This method extracts a set of frames around a given time point, according to the specified window size and
	 * placement, and performs an intensity projection (e.g., max, min, sum, standard deviation) across those frames
	 * to increase contrast and exposure time.
	 *
	 * @param imp ImagePlus input
	 * @param typeOfProjection the type of projection to perform; accepted values include "max", "min", "sum", "sd", etc
	 * @param window_place determines how the projection window is placed relative to the current time frame;
	 *  *                  accepted values are "left", "middle", or "right"
	 * @param window number of frames/2 that we want to be projected
	 * @return ImageProcessor resulting from the temporal projection over the selected window
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

		return projection;
	}


	/**
	 * This method performs an intensity projection across all frames of the input. In our case, it is used to highlight
	 * microtubules' trajectories over the whole time series.
	 *
	 * @param imp ImagePlus input
	 * @param typeOfProjection the type of projection to perform; accepted values include "max", "min", "sum", "sd", etc
	 * @return ImagePlus resulting from the intensity projection across all frames
	 */
	private ImagePlus totalProjection(ImagePlus imp, String typeOfProjection){
		return ZProjector.run(imp,typeOfProjection);
	}


	/**
	 * For each time frame, this method computes the cumulative difference between the current frame and a window of
	 * previous frames of a chosen size. In our case, each resulting slice represents the fluorescent protein motion over
	 * a chosen time-lapse.
	 *
	 * @param imp ImagePlus input
	 * @param title the title of the resulting projected image
	 * @param windowSize the number of previous frames to include in the difference calculation
	 * @return ImagePlus with teh temporal difference for each frame
	 */
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
