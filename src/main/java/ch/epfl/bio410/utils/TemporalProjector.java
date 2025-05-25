package ch.epfl.bio410.utils;


import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.ZProjector;
import ij.process.ImageProcessor;


/**
 * The TemporalProjector class applies temporal projections to time-lapse image stacks.
 * The TemporalProjector implements Function<ImageProcessor, ImageProcessor>, allowing it to be used with stack-processing
 * utilities that expect per-slice processing.
 * It performs a projection (e.g., "max", "min", "avg", etc.) over a temporal window around each frame, creating a new
 * image where each slice is a projection of its temporal neighborhood.
 */
public class TemporalProjector implements java.util.function.Function<ImageProcessor,ImageProcessor> {
    private final ImagePlus imp;          // the full movie
    private final String projType;     // "max", "min", "avg", …
    private final String windowPlace;  // "middle", "left", "right"
    private final int windowSize;   // in frames
    private int currentFrame = 0; // will be updated externally

    /**
     * Constructor of the class.
     *
     * @param imp The input ImagePlus object.
     * @param projType The type of temporal projection to apply ("max", "min", "avg", etc.).
     * @param windowPlace The placement of the temporal window relative to the current frame:
     * "middle" (centered), "left" (preceding frames), or "right" (following frames).
     * @param windowSize The number of frames included in the temporal projection window.
     */
    public TemporalProjector(ImagePlus imp, String projType, String windowPlace, int windowSize) {
        this.imp         = imp;
        this.projType    = projType;
        this.windowPlace = windowPlace;
        this.windowSize  = windowSize;
    }

    /**
     * This method applies the temporal projection to the next frame in the stack.
     * This method is called automatically by "processStack()". The ip argument is ignored
     * since the real image data is accessed directly from the ImagePlus reference.
     *
     * @param ip Ignored input frame (used only to advance internal frame count).
     * @return An ImageProcessor representing the temporally projected frame.
     */
    @Override
    public ImageProcessor apply(ImageProcessor ip) {
        ++currentFrame;
        return projectFrame(currentFrame);
    }

    /**
     * This method performs the temporal projection over a window of frames centered (or offset) around frame t.
     *
     * @param t The index of the current frame.
     * @return An ImageProcessor containing the projection result over the defined window.
     */
    private ImageProcessor projectFrame(int t) {
        int nFrames = imp.getNFrames();
        int start, end;

        // determine start and end frame indices for the projection window
        if ("middle".equals(windowPlace)) {
            start = Math.max(1, t - windowSize / 2);
            end   = Math.min(nFrames, t + windowSize / 2);
        } else if ("left".equals(windowPlace)) {
            start = Math.max(1, t - windowSize);
            end   = t;
        } else { // right
            start = t;
            end   = Math.min(nFrames, t + windowSize);
        }

        // create a temporary sub-stack of frames in the window
        ImageStack sub = new ImageStack(imp.getWidth(), imp.getHeight());
        for (int f = start; f <= end; f++) {
            imp.setPosition(1, 1, f);
            sub.addSlice(imp.getProcessor().duplicate());
        }
        ImagePlus tmp = new ImagePlus("sub", sub);
        ImageProcessor proj = ZProjector.run(tmp, projType).getProcessor();
        tmp.close();

        return proj;
    }
}
