/**
 * Copyright 2017 Chuck Wolber
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy 
 * of this software and associated documentation files (the "Software"), to deal 
 * in the Software without restriction, including without limitation the rights 
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
 * copies of the Software, and to permit persons to whom the Software is 
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package bdl;

import java.security.SecureRandom;
import java.util.Random;

/**
 *
 * @author chuckwolber
 */
public enum RandomWeight
{
    INSTANCE;
    
    private Random _random;
    
    public double nextDouble(double origin, double bound) {
        if (_random == null)
            initRandom();
        double r = _random.nextDouble();
        r = r * (bound - origin) + origin;
        if (r >= bound)
            r = Math.nextDown(bound);
        return r;
    }
    
    private void initRandom() {
        byte[] seed = SecureRandom.getSeed(8);
        long lSeed = 0;
        for (int i=0; i<seed.length; i++)
            lSeed |= ((0xff & seed[i]) << (seed.length-1-i)*8);
        _random = new Random(lSeed);
    }
}
