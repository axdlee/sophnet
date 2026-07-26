# Privacy Policy

Last updated: July 25, 2026

This policy describes how the SophNet Dify plugin processes data.

## Data processed

Depending on the selected model, the plugin sends the following data to SophNet to perform the requested inference:

- SophNet API Key, Project ID, and configured EasyLLM ID
- Prompts, conversation messages, tool definitions, and model parameters
- Text submitted for embedding or text-to-speech
- Audio submitted for speech-to-text
- Images, video, or audio included in supported multimodal prompts

## Purpose and storage

The data is used only to authenticate with SophNet and fulfill model inference requests. The plugin does not create its own persistent storage for credentials, prompts, media, model responses, or generated audio. Dify and SophNet may process or retain data according to their own deployment settings and policies.

## Third-party service

The plugin communicates with [SophNet](https://www.sophnet.com/) over HTTPS. SophNet receives the request data described above and returns model output or task status. Review SophNet's current service terms and privacy policy before using the plugin with sensitive data.

## Security

Credentials are passed to SophNet in authorization headers and are not included in plugin error messages. Users are responsible for limiting API key permissions, rotating credentials, and securing their Dify deployment.

## User choices

You can stop data processing by removing the SophNet credentials from Dify or uninstalling the plugin. Requests already transmitted to SophNet are governed by SophNet's applicable policies.

## Contact

For privacy questions, open an issue at <https://github.com/axdlee/sophnet/issues>.
