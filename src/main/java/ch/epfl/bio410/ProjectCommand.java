package ch.epfl.bio410;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.io.OpenDialog;
import ij.plugin.GaussianBlur3D;
import ij.plugin.ImageCalculator;
import ij.process.ImageProcessor;
import net.imagej.ImageJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;


@Plugin(type = Command.class, menuPath = "Plugins>BII>Project Template")
public class ProjectCommand implements Command {

	public void run() {
		IJ.log("Hello World");

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

		new ImagePlus("DoG Stack", outputStack).show();

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
