#version 450

layout( location = 0 ) in vec3 aVertexPosition;
layout( location = 1 ) in vec3 aVertexNormal;

out vec3 aVertexPositionOut;
out vec3 anAdaptedVertexNormal;
out vec3 vs_sourceLight;
uniform mat4 uMVPMatrix; //  View * Model
uniform mat4 uMVMatrix; //  Projection
uniform vec3 uCamPos; //  View * Model

void main()
{

	vs_sourceLight = vec3( uMVPMatrix * vec4(uCamPos, 1.f) );

	aVertexPositionOut = vec3( uMVMatrix * vec4(aVertexPosition, 1.f) );

	gl_Position = uMVPMatrix*vec4( aVertexPosition, 1.f );
	
	anAdaptedVertexNormal =  aVertexNormal ;
	
}
