#version 450

layout( location = 0 ) out vec4 fragColor;


in vec3 anAdaptedVertexNormal;
in vec3 aVertexPositionOut;
in vec3 vs_sourceLight;

void main()
{
	vec3 color=vec3(1.0,0.0,0.0);

	vec3 lightDir =  vec3(0.0, 0.0, 3.0);

	vec3 normal=anAdaptedVertexNormal;

	vec3 Li = normalize( vs_sourceLight - aVertexPositionOut );	//vecteur de la lumière émise

	if( dot(Li, normal) < 0 ){
		normal *= -1;
	}

	vec3 eclairage_diffus = max ( dot( normalize(anAdaptedVertexNormal) , Li ), 0.f ) * color;


	fragColor = vec4( anAdaptedVertexNormal, 0.f );
	

}
