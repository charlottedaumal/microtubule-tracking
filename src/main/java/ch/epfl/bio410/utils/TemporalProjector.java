package ch.epfl.bio410.utils;

import ij.ImagePlus;
import ij.ImageStack;
import ij.plugin.ZProjector;
import ij.process.ImageProcessor;

/**  Wraps a temporal projection so it looks like
 *   ImageProcessor -> ImageProcessor (suitable for processStack).
 */
public class TemporalProjector implements java.util.function.Function<ImageProcessor,ImageProcessor>
{
    private final ImagePlus imp;          // the full movie
    private final String    projType;     // "max", "min", "avg", …
    private final String    windowPlace;  // "middle", "left", "right"
    private final int       windowSize;   // in frames
    private       int       currentFrame = 0; // will be updated externally

    public TemporalProjector(ImagePlus imp,
                             String projType,
                             String windowPlace,
                             int    windowSize)
    {
        this.imp         = imp;
        this.projType    = projType;
        this.windowPlace = windowPlace;
        this.windowSize  = windowSize;
    }

    /** Called by processStack; ip is ignored except to advance the frame index. */
    @Override
    public ImageProcessor apply(ImageProcessor ip)
    {
        ++currentFrame;                          // 1 … nFrames
        return projectFrame(currentFrame);       // do the real work
    }

    /* -------- real projection, returns one slice -------- */
    private ImageProcessor projectFrame(int t)
    {
        int nFrames = imp.getNFrames();
        int start, end;

        if ("middle".equals(windowPlace)) {
            start = Math.max(1, t - windowSize / 2);
            end   = Math.min(nFrames, t + windowSize / 2);
        } else if ("left".equals(windowPlace)) {
            start = Math.max(1, t - windowSize);
            end   = t;
        } else { /* "right" */
            start = t;
            end   = Math.min(nFrames, t + windowSize);
        }

        /* build tiny sub-stack */
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
