#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;
uniform vec4 colDiffuse;

// Output fragment color
out vec4 finalColor;

// NOTE: Add here your custom variables

// NOTE: Render size values should be passed from code
const float renderWidth = 800;
const float renderHeight = 450;

void main()
{
    vec2 texSize = vec2(renderWidth, renderHeight);
    vec2 tc = fragTexCoord*texSize;

    vec4 color = texture2D(texture0, tc/texSize)*colDiffuse*fragColor;

    finalColor = vec4(color.rrr, 1.0);
}
