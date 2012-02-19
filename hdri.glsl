#line 1 "hdri.glsl"
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
    vec3 XYZ = RGB_to_XYZ * texCol.xyz;
    float x = XYZ.x / (XYZ.x + XYZ.y + XYZ.z);
    float y = XYZ.y / (XYZ.x + XYZ.y + XYZ.z);
    float Y = XYZ.y;
    // and tone map the luminosity Y, before transforming back:
    if(Y != 0.0)
    {
        Y = pow(Y, hdriPow);
        Y = Y / (hdriExposure + Y);
    }
    //Y = clamp(Y/100.0, 0.0, 1.0);
    XYZ = vec3(Y*x/y, Y, Y*(1.0-x-y)/y);
    vec3 RGB = XYZ_to_RGB * XYZ;
    gl_FragColor = vec4(sRGB_gamma_correct(RGB.x),
                        sRGB_gamma_correct(RGB.y),
                        sRGB_gamma_correct(RGB.z),
                        texCol.w);
}
