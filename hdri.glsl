// Copyright (C) 2011, Chris Foster and the other authors and contributors.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of the software's owners nor the names of its
//   contributors may be used to endorse or promote products derived from this
//   software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// (This is the New BSD license)
#line 29 "hdri.glsl"

// Convert linear RGB component to sRGB gamma corrected color space
float sRGB_gamma_correct(float c)
{
    const float a = 0.055;
    if(c < 0.0031308)
        return 12.92*c;
    else
        return (1.0+a)*pow(c, 1.0/2.4) - a;
}

uniform sampler2D tex;
uniform float hdriExposure;
uniform float hdriPow;
uniform bool whiteBackground;

// HDRI tone mapping
void main()
{
    // RGB->XYZ transform for standard sRGB linear RGB
    const mat3 RGB_to_XYZ = mat3(
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    );
    const mat3 XYZ_to_RGB = mat3(
         3.2404542,-1.5371385,-0.4985314,
        -0.9692660, 1.8760108, 0.0415560,
         0.0556434,-0.2040259, 1.0572252
    );
    // Treat input texture as linear color space.  We transform to
    // CIE xyY color space
    vec4 texCol = texture2D(tex, gl_TexCoord[0].xy);
    vec3 RGB = vec3(whiteBackground ? 1.0 : 0.0);
    if(texCol.x != 0.0 || texCol.y != 0.0 || texCol.z != 0.0)
    {
        vec3 XYZ = RGB_to_XYZ * texCol.xyz;
        float x = XYZ.x / (XYZ.x + XYZ.y + XYZ.z);
        float y = XYZ.y / (XYZ.x + XYZ.y + XYZ.z);
        float Y = XYZ.y;
        // and tone map the luminosity Y, before transforming back:
        Y = hdriExposure*pow(Y, hdriPow);
        Y = Y / (1.0 + Y);
        if(whiteBackground)
            Y = 1.0 - Y;
        //Y = clamp(Y/100.0, 0.0, 1.0);
        XYZ = vec3(Y*x/y, Y, Y*(1.0-x-y)/y);
        RGB = XYZ_to_RGB * XYZ;
    }
    gl_FragColor = vec4(sRGB_gamma_correct(RGB.x),
                        sRGB_gamma_correct(RGB.y),
                        sRGB_gamma_correct(RGB.z),
                        texCol.w);
}
