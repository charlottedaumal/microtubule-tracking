package ch.epfl.bio410.utils;

import ij.ImagePlus;
import ij.process.Blitter;
import ij.process.ImageProcessor;

import java.util.function.Function;

/**  Returns Δ-frame = current − preceding N frames (running sum of
 *   subtractions).  Implements Function<ImageProcessor,ImageProcessor>
 *   so it can be passed straight into processStack.
 */
public class TemporalDifferencer
        implements Function<ImageProcessor, ImageProcessor>
{
    private final ImagePlus imp;   // full movie (not duplicated)
    private final int       window;
    private       int       frameIdx = 0;   // will advance automatically

    public TemporalDifferencer(ImagePlus imp, int window)
    {
        this.imp    = imp;
        this.window = window;
    }

    @Override
    public ImageProcessor apply(ImageProcessor ipIgnored)
    {
        ++frameIdx;                              // 1 … nFrames
        int t        = frameIdx;
        int nFrames  = imp.getNFrames();
        int start    = Math.max(1, t - window);  // left-hand window limit

        /* ---- copy current frame ---- */
        imp.setPosition(1, 1, t);
        ImageProcessor delta = imp.getProcessor().duplicate();

        /* ---- subtract previous frames in the window ---- */
        for (int i = start; i < t; i++) {
            imp.setPosition(1, 1, i);
            ImageProcessor prev = imp.getProcessor();   // no duplicate needed
            delta.copyBits(prev, 0, 0, Blitter.SUBTRACT);
        }
        return delta;                       // one processed slice
    }
}
