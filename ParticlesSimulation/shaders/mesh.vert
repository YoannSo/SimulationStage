#version 450

layout( location = 0 ) in vec3 aVertexPosition;
layout( location = 1 ) in vec3 aVertexNormal;

out vec3 aVertexPositionOut;
out vec3 anAdaptedVertexNormal;
out vec3 vs_sourceLight;
uniform mat4 uMVPMatrix; //  View * Model
uniform mat4 uRotMatrix; //  View * Model
uniform mat4 uTransfoMatrix;
uniform mat4 uMVMatrix; //  MODEL VIEWs
uniform vec4 uCamPos; //  View * Model

void main()
{
	vec4 camPos=uCamPos*uRotMatrix;
	vs_sourceLight = vec3( uMVMatrix * vec4(0.f,0.f,0.f, 1.f) );

	aVertexPositionOut = vec3( uMVPMatrix* vec4(aVertexPosition, 1.f) );

	gl_Position = uMVPMatrix*vec4( aVertexPosition, 1.f );
	
	anAdaptedVertexNormal =  aVertexNormal ;
	
}
