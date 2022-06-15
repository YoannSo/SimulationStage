#version 450

layout( location = 0 ) in vec3 aVertexPosition;
layout( location = 1 ) in vec3 aVertexNormal;


uniform mat4 uMVPMatrix; // Projection * View * Model

void main()
{
	gl_Position = vec4( 10.f,10.f,10.f, 1.f );
}
