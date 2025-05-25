package ch.epfl.bio410.utils;


import ij.ImagePlus;
import ij.process.Blitter;
import ij.process.ImageProcessor;

import java.util.function.Function;


/**
 * The TemporalDifferencer class implements a Function<ImageProcessor, ImageProcessor>,
 * allowing it to be used in image stack processing pipelines.
 * For each time frame t, this class computes a temporal difference image by subtracting a given preceding number of frames
 * from the current frame t. Here, this is useful to get single leading spots for each trajectory in the current frame t.
 */
public class TemporalDifferencer implements Function<ImageProcessor, ImageProcessor> {
    private final ImagePlus imp;   // full movie (not duplicated)
    private final int       window;
    private       int       frameIdx = 0;   // will advance automatically

    /**
     * Constructor of the class.
     *
     * @param imp The ImagePlus object representing the image stack.
     * @param window The number of preceding frames to subtract from the current frame.
     */
    public TemporalDifferencer(ImagePlus imp, int window) {
        this.imp    = imp;
        this.window = window;
    }

    /**
     * This method applies the temporal differencing operation to the current frame in the sequence. It is automatically
     * called for each frame in the stack during processing.
     * It updates the internal frame index, retrieves the current frame, and subtracts the pixel values of the previous
     * "window" frames from it.
     *
     * @param ipIgnored The ignored ImageProcessor input (used only to advance internal frame count).
     * @return A new ImageProcessor containing the result of the temporal differencing.
     */
    @Override
    public ImageProcessor apply(ImageProcessor ipIgnored) {
        ++frameIdx;
        int t = frameIdx;
        int start = Math.max(1, t - window);  // left-hand window limit

        // copy current frame
        imp.setPosition(1, 1, t);
        ImageProcessor delta = imp.getProcessor().duplicate();

        // subtract previous frames in the window
        for (int i = start; i < t; i++) {
            imp.setPosition(1, 1, i);
            ImageProcessor prev = imp.getProcessor();   // no duplicate needed
            delta.copyBits(prev, 0, 0, Blitter.SUBTRACT);
        }

        return delta; // one processed slice
    }
}
