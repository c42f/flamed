#line 1 "point_accum.glsl"
#version 120

void main()
{
    float r = 2*length(gl_PointCoord - 0.5);
    float a = 1 - smoothstep(0, 1, r);
    gl_FragColor = vec4(gl_Color.x, gl_Color.y, gl_Color.z, a);
}
