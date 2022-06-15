#version 450

layout( location = 0 ) out vec4 fragColor;

void main()
{
 const vec3 lightDir = vec3(0.577, 0.577, 0.577);

      vec3 N;
    N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

	fragColor = vec4( 1.f, 0.f, 0.f, 1.f );
}
