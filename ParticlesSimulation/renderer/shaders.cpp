

#define STRINGIFY(A) #A
 // vertex shader
const char* vertexShader = STRINGIFY(
    uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels
uniform float densityScale;
uniform float densityOffset;
uniform float colorMode;
void main()
{
    // calculate window-space point size

    vec3 posEye = vec3(gl_ModelViewMatrix * vec4(gl_Vertex.xyz, 1.0));
    float dist = length(posEye);
    gl_PointSize = pointRadius * (pointScale / dist);
    gl_TexCoord[0] = gl_MultiTexCoord0;
    
    gl_Position = gl_ModelViewProjectionMatrix * vec4(gl_Vertex.xyz, 1.0);


<<<<<<< HEAD
    if (colorMode ==1) {
=======
    if (colorMode == 1) {
>>>>>>> f77b5d5a58d4ac1b8aa7286792f06f1c8c2a9e60
        gl_FrontColor = vec4(1.f, gl_Vertex.y+0.3f, 0.f, 0.f);

    }
    else if (colorMode == -1) {
        gl_FrontColor = gl_Color;

    }

}
);

// pixel shader for rendering points as shaded spheres
const char* spherePixelShader = STRINGIFY(
    void main()
{
    const vec3 lightDir = vec3(0.577, 0.577, 0.577);

    // calculate normal from texture coordinates
    vec3 N;
    N.xy = gl_TexCoord[0].xy * vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(N.xy, N.xy);

    if (mag > 1.0) discard;   // kill pixels outside circle

    N.z = sqrt(1.0 - mag);

    // calculate lighting
    float diffuse = max(0.0, dot(lightDir, N));

    gl_FragColor = gl_Color * diffuse;
}
);
//-----------------------------------------------------------------------------------------
const char* meshVertex = STRINGIFY(

uniform mat4 uMVPMatrix; // Projection * View * Model

void main()
{
    gl_Position = uMVPMatrix * vec4(gl_Vertex.xyz, 1.0);   
    gl_FrontColor = vec4(1.f, gl_Vertex.y + 1, 0.f, 0.f);

}
);
// pixel shader for rendering points as shaded spheres
const char* meshFrag = STRINGIFY(

    void main()
{
    gl_FragColor = gl_Color;
}
);
