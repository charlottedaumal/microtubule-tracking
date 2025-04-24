package ch.epfl.bio410;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.OpenDialog;
import ij.plugin.GaussianBlur3D;
import ij.plugin.ImageCalculator;
import ij.plugin.ZProjector; // for median filtering
import ij.process.Blitter; // for median filtering
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;


@Plugin(type = Command.class, menuPath = "Plugins>BII>Microtubule Gang")
public class ProjectCommand implements Command {

	public void run() {

		// Prompt the user to select a file
		OpenDialog fileChooser = new OpenDialog("Select a file", null);

		String filePath = fileChooser.getDirectory();
		String fileName = fileChooser.getFileName();
		
		ImagePlus imp = IJ.openImage(filePath+fileName);

		imp.show();

		int nFrames = imp.getNFrames();
		ImageStack outputStack = new ImageStack(imp.getWidth(), imp.getHeight());

		for (int t = 0; t < nFrames; t++) {
			imp.setPosition(1, 1, 1 + t);
			ImageProcessor dog = dog(imp.getProcessor(), 5, 1.25);
			outputStack.addSlice(dog);
		}
		//new ImagePlus("DoG Stack", outputStack).show();

		// Enhance contrast on DoG image
		ImagePlus dogProcessedImp = new ImagePlus("Processed DoG Stack", outputStack);
		IJ.run(dogProcessedImp, "Enhance Contrast", "saturated=0.1 normalize process_all");
		dogProcessedImp.show();

		// Apply median filter
		ImagePlus medianImp = median(imp, "Median Stack");
		// medianImp.show();

		// Enhance contrast on median filtered image
		ImagePlus medianProcessedImp = medianImp.duplicate();
		medianProcessedImp.setTitle("Processed Median Stack");
		IJ.run(medianProcessedImp, "Enhance Contrast", "saturated=0.1 normalize process_all");
		medianProcessedImp.show();

		// Apply dog filter on median filtered image
		ImageStack dogMedianStack = new ImageStack(medianImp.getWidth(), medianImp.getHeight());
		for (int t = 0; t < medianImp.getNSlices(); t++) {
			medianImp.setPosition(1, t+1, t+1);
			ImageProcessor dog2 = dog(medianImp.getProcessor(), 5, 1.25);
			dogMedianStack.addSlice(dog2);
		}
		// new ImagePlus("DoG on Median Stack", dogMedianStack).show();

		// Enhance contrast on DoG and Median filtered image
		ImagePlus dogMedianImp = new ImagePlus("Processed DoG on Median Stack", dogMedianStack);
		IJ.run(dogMedianImp, "Enhance Contrast", "saturated=0.1 normalize process_all");
		dogMedianImp.show();

		// Apply median filter with simple command
		ImagePlus median2Imp = imp.duplicate();
		median2Imp.setTitle("Median 2 Stack");
		IJ.run(median2Imp, "Median...", "radius=15 stack");
		ImagePlus medianProcessed2Imp = ImageCalculator.run(imp, median2Imp, "Subtract create stack");
		//IJ.run(medianProcessed2Imp, "Enhance Contrast", "saturated=0.1 normalize process_all");
		medianProcessed2Imp.setTitle("Processed Median Stack 2");
		//medianProcessed2Imp.show();

		// Apply dog filter on median filtered image 2
		ImageStack dogMedian2Stack = new ImageStack(medianProcessed2Imp.getWidth(), medianProcessed2Imp.getHeight());
		for (int t = 0; t < medianProcessed2Imp.getNFrames(); t++) {
			medianProcessed2Imp.setPosition(1, 1, t+1);
			ImageProcessor dog3 = dog(medianProcessed2Imp.getProcessor(), 5, 1.25);
			dogMedian2Stack.addSlice(dog3);
		}
		//new ImagePlus("DoG on Median Stack 2", dogMedian2Stack).show();

		// Enhance contrast on DoG and Median filtered image 2
		ImagePlus dogMedian2Imp = new ImagePlus("Processed DoG on Median Stack 2", dogMedian2Stack);
		IJ.run(dogMedian2Imp, "Enhance Contrast", "saturated=0.1 normalize process_all");
		dogMedian2Imp.show();

	}


	private ImageProcessor dog(ImageProcessor ip, double sigma1, double sigma2 ) {
		ImagePlus g1 = new ImagePlus("g1", ip.duplicate());
		ImagePlus g2 = new ImagePlus("g2", ip.duplicate());
//		double sigma2 = (Math.sqrt(2) * sigma);
		GaussianBlur3D.blur(g1, sigma1, sigma1, 0);
		GaussianBlur3D.blur(g2, sigma2, sigma2, 0);
		ImagePlus dog = ImageCalculator.run(g2, g1, "Subtract create stack");
		return dog.getProcessor();
	}


	// Temporal median projection computation... need to work a bit more on this to be sure
	// it is what we want (not sure about that after reading some info, it might be better to have
	// a temporal median filtering (with a time window radius)
	private ImagePlus median(ImagePlus imp, String title) {
		ImageStack originalStack = imp.getStack().duplicate();
		ImageStack medianStack = new ImageStack(imp.getWidth(), imp.getHeight());

		// Generate the median projection
		ZProjector zp = new ZProjector(imp);
		zp.setMethod(ZProjector.MEDIAN_METHOD);
		zp.doProjection();
		ImagePlus medianProjection = zp.getProjection();
		ImageProcessor medianProc = medianProjection.getProcessor();

		for (int i = 1; i <= imp.getStackSize(); i++) {
			ImageProcessor slice = originalStack.getProcessor(i).duplicate();
			slice.copyBits(medianProc, 0, 0, Blitter.SUBTRACT);
			medianStack.addSlice("slice-" + i, slice);
		}

		return new ImagePlus(title, medianStack);
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
